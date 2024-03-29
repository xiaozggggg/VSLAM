#include "global_fusion.h"

GlobalOptimization::GlobalOptimization()
{
    mbInitGPS = false;
    mbGPSCtrl = false;
    mbThreadCtrl = false;

    WGPS_T_WVIO = Eigen::Matrix4d::Identity();
    optThread = std::thread(&GlobalOptimization::optimize, this);
}

GlobalOptimization::~GlobalOptimization()
{
    mbThreadCtrl = true;
    optThread.join();
}

// TODO
// One thing to note is that ‘posAccuracy’ represents the covariance information output by GPS.
// If you use RTK later, you need to modify this function according to the actual situation
void GlobalOptimization::InputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy)
{
    double xyz[3];
    GPS2XYZ(latitude, longitude, altitude, xyz);
    std::vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
    GPSPositionMap[t] = tmp; // t is the timeStamp
    mbGPSCtrl = true;
}

// This is the odometer input function, the odometer comes from the
// positioning information output of orb_slam3
void GlobalOptimization::InputOdom(double t, Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ)
{
    mtxOdm.lock();

    std::vector<double> localPose{OdomP.x(), OdomP.y(), OdomP.z(), OdomQ.w(), OdomQ.x(), OdomQ.y(), OdomQ.z()};
    localPoseMap[t] = localPose;

    Eigen::Quaterniond globalQ;
    globalQ = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomQ;
    Eigen::Vector3d globalP = WGPS_T_WVIO.block<3, 3>(0, 0) * OdomP + WGPS_T_WVIO.block<3, 1>(0, 3);
    std::vector<double> globalPose{globalP.x(), globalP.y(), globalP.z(),
                                   globalQ.w(), globalQ.x(), globalQ.y(), globalQ.z()};
    globalPoseMap[t] = globalPose;

    lastP = globalP;
    lastQ = globalQ;

    mtxOdm.unlock();

    // geometry_msgs::PoseStamped pose_stamped;
    // pose_stamped.header.stamp = ros::Time(t);
    // pose_stamped.header.frame_id = "world_vio";
    // pose_stamped.pose.position.x = lastP.x();
    // pose_stamped.pose.position.y = lastP.y();
    // pose_stamped.pose.position.z = lastP.z();
    // pose_stamped.pose.orientation.x = lastQ.x();
    // pose_stamped.pose.orientation.y = lastQ.y();
    // pose_stamped.pose.orientation.z = lastQ.z();
    // pose_stamped.pose.orientation.w = lastQ.w();
    // global_path.header = pose_stamped.header;
    // global_path.poses.push_back(pose_stamped);
}

void GlobalOptimization::GetGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ)
{
    odomP = lastP;
    odomQ = lastQ;
}

void GlobalOptimization::GPS2XYZ(double latitude, double longitude, double altitude, double *xyz)
{
    if (!mbInitGPS)
    {
        // The GPS coordinates of the first frame will be initialized
        // to (0,0,0) after the Reset function
        geoConverter.Reset(latitude, longitude, altitude);
        mbInitGPS = true;
    }
    // Convert GPS global coordinates to plane coordinates
    geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);

    // exchange X-Axis and Y-Axis.
    double tmp_value = xyz[0];
    xyz[0] = xyz[1];
    xyz[1] = -tmp_value;
}

void GlobalOptimization::optimize()
{
    while (true)
    {
        if (mbGPSCtrl)
        {
            mbGPSCtrl = false;

            ceres::Problem problem;

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            // options.minimizer_progress_to_stdout = true;
            // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;

            ceres::Solver::Summary summary;

            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(1.0);

            ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

            // Add parameters
            mtxOdm.lock();
            int length = localPoseMap.size();
            double t_array[length][3];
            double q_array[length][4];
            std::map<double, std::vector<double>>::iterator iter;
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                t_array[i][0] = iter->second[0];
                t_array[i][1] = iter->second[1];
                t_array[i][2] = iter->second[2];
                q_array[i][0] = iter->second[3];
                q_array[i][1] = iter->second[4];
                q_array[i][2] = iter->second[5];
                q_array[i][3] = iter->second[6];
                problem.AddParameterBlock(q_array[i], 4, local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
            }

            std::map<double, std::vector<double>>::iterator iterVIO, iterVIONext, iterGPS;
            int i = 0;
            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
            {
                iterVIONext = iterVIO;
                iterVIONext++;
                if (iterVIONext != localPoseMap.end())
                {
                    Eigen::Matrix4d wTi = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d wTj = Eigen::Matrix4d::Identity();

                    wTi.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIO->second[3], iterVIO->second[4], iterVIO->second[5], iterVIO->second[6]).toRotationMatrix();
                    wTi.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIO->second[0], iterVIO->second[1], iterVIO->second[2]);
                    wTj.block<3, 3>(0, 0) = Eigen::Quaterniond(iterVIONext->second[3], iterVIONext->second[4], iterVIONext->second[5], iterVIONext->second[6]).toRotationMatrix();
                    wTj.block<3, 1>(0, 3) = Eigen::Vector3d(iterVIONext->second[0], iterVIONext->second[1], iterVIONext->second[2]);
                    Eigen::Matrix4d iTj = wTi.inverse() * wTj;
                    Eigen::Quaterniond iQj;
                    iQj = iTj.block<3, 3>(0, 0);
                    Eigen::Vector3d iPj = iTj.block<3, 1>(0, 3);

                    ceres::CostFunction *vio_function = RelativeRTError::Create(iPj.x(), iPj.y(), iPj.z(), iQj.w(), iQj.x(), iQj.y(), iQj.z(), 0.1, 0.01);
                    problem.AddResidualBlock(vio_function, NULL, q_array[i], t_array[i], q_array[i + 1], t_array[i + 1]);
                }

                double t = iterVIO->first;
                iterGPS = GPSPositionMap.find(t);
                if (iterGPS != GPSPositionMap.end())
                {
                    ceres::CostFunction *gps_function = TError::Create(iterGPS->second[0], iterGPS->second[1], iterGPS->second[2], iterGPS->second[3]);
                    problem.AddResidualBlock(gps_function, loss_function, t_array[i]);
                }
            }
            ceres::Solve(options, &problem, &summary);

            // Update global pose
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                std::vector<double> globalPose{t_array[i][0], t_array[i][1], t_array[i][2], q_array[i][0], q_array[i][1], q_array[i][2], q_array[i][3]};
                iter->second = globalPose;
                if (i == length - 1)
                {
                    Eigen::Matrix4d WVIO_T_body = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d WGPS_T_body = Eigen::Matrix4d::Identity();
                    double t = iter->first;
                    WVIO_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(localPoseMap[t][3], localPoseMap[t][4], localPoseMap[t][5], localPoseMap[t][6]).toRotationMatrix();
                    WVIO_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(localPoseMap[t][0], localPoseMap[t][1], localPoseMap[t][2]);
                    WGPS_T_body.block<3, 3>(0, 0) = Eigen::Quaterniond(globalPose[3], globalPose[4], globalPose[5], globalPose[6]).toRotationMatrix();
                    WGPS_T_body.block<3, 1>(0, 3) = Eigen::Vector3d(globalPose[0], globalPose[1], globalPose[2]);
                    WGPS_T_WVIO = WGPS_T_body * WVIO_T_body.inverse();
                }
            }
            mtxOdm.unlock();
        }

        if (mbThreadCtrl)
            break;
    }
}