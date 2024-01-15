/* ============================================================================
 * Copyright (c) 2023 ShangZhi.
 * All rights reserved.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted.
 *
 * THE SOFTWARE IS PROVIDED 'AS IS' AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 * ============================================================================
 */

#include "global_fusion.h"
#include <queue>
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Header.h>
#include <std_msgs/Bool.h>
#include <yaml-cpp/yaml.h>
#include <sensor_msgs/NavSatFix.h>
#include <message_filters/subscriber.h>
#include <fstream>
#include "util/GpsPosition.h"
#include "util/Vslam.h"
#include "mower_msgs/VslamState.h"

#include "common.h"
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

int counter = 0;

class FusionNode
{
public:
    FusionNode() : nh_("~")
    {
        node_name_ = ros::this_node::getName();
        nh_.param<std::string>(node_name_ + "/settings_file", setting_file_, "file_not_set");
        nh_.param<std::string>(node_name_ + "/rtk_topic", rtk_topic_, "/nanobot/gpsposition");
        nh_.param<std::string>(node_name_ + "/vslam_topic", vslam_topic_, "/orb_slam3_ros/camera_pose");
        nh_.param<std::string>(node_name_ + "/fusion_topic", publish_topic_, "/mower/vslam_pose");
        nh_.param<std::string>(node_name_ + "/state_topic", vslam_state_, "/mower/vslam_state");
        nh_.param<bool>(node_name_ + "/save_csv_flag", save_csv_, false);

        ParameterInit();

        // Subscribe and publish topics.
        gps_sub_ = nh_.subscribe(rtk_topic_, 1, &FusionNode::GpsCallBack, this);
        vslam_sub_ = nh_.subscribe(vslam_topic_, 1, &FusionNode::VioCallBack, this);
        vslam_init_sub_ = nh_.subscribe("/vslam/init_flag", 1, &FusionNode::InitCallBack, this);

        fusion_pub_ = nh_.advertise<util::Vslam>(publish_topic_, 1);
        // Post self-test results topic.
        self_check_pub_ = nh_.advertise<mower_msgs::VslamState>(vslam_state_, 1);

        // Set a timer_ to check self-test conditions regularly.
        timer_ = nh_.createTimer(ros::Duration(1.0), &FusionNode::SelfCheckTimerCallback, this);
    }

    void ParameterInit()
    {
        // Read the working mode from yaml file.
        cv::FileStorage fsSettings(setting_file_.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            ROS_WARN("Failed to open settings file");
            exit(-1);
        }
        cv::FileNode node = fsSettings["System.WorkingMode"];
        if (!node.empty())
        {
            if (node.real() != 0)
            {
                location_mode_ = true;
                ROS_INFO("<--- Fusion node: location mode! --->");
            }
            else
            {
                location_mode_ = false;
                ROS_INFO("<--- Fusion node: create map mode! --->");
            }
        }
        else
        {
            location_mode_ = false;
            ROS_INFO("<--- Fusion node: create map mode! --->");
        }

        // Initialize flag variable.
        gps_state_ = false;
        vio_state_ = false;
        fusion_state_ = false;
        vslam_init_flag_ = false;

        gps_ctr_flag_ = false;

        // Set the flag variable for self-test success/failure
        self_check_result_.data = false;

        last_vio_time_ = -1;

        map_frame_id_ = "map";
        car_frame_id_ = "base_link";

        read_gauss_x_ = 0.0;
        read_gauss_y_ = 0.0;
        pitch_ = 0.0;
        roll_ = 0.0;
        azimuth_ = 0.0;

        transformationMatrix_ = Eigen::Matrix4d::Identity();
    }

    // InitCallBack function.
    void InitCallBack(const std_msgs::Bool::ConstPtr &msg)
    {
        vslam_init_flag_ = msg->data;
        ROS_INFO("<---- Vslam init successful! ---->");
    }

    // GPS callback function.
    void GpsCallBack(const util::GpsPositionConstPtr &gps_msg)
    {
        gps_state_ = true;
        gps_mutex_.lock();
        gps_queue_.push(gps_msg);
        gps_mutex_.unlock();

        if (!gps_ctr_flag_)
        {
            double gauss_x = gps_msg->gaussX / 100.0;
            double gauss_y = gps_msg->gaussY / 100.0;

            if (!location_mode_)
            {
                SaveGauss(gauss_x, gauss_y);
            }
            else
            {
                if (!ReadGauss())
                    ROS_WARN("GPS parameters read failure!");
                ComputeMatrix(gauss_x, gauss_y);
            }
            gps_ctr_flag_ = true;
        }
    }

    // VSLAM callback function.
    void VioCallBack(const nav_msgs::Odometry::ConstPtr &pose_msg)
    {
        vio_state_ = true;

        last_vio_time_ = pose_msg->header.stamp.toSec();

        Eigen::Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
        Eigen::Quaterniond vio_q;
        vio_q.w() = pose_msg->pose.pose.orientation.w;
        vio_q.x() = pose_msg->pose.pose.orientation.x;
        vio_q.y() = pose_msg->pose.pose.orientation.y;
        vio_q.z() = pose_msg->pose.pose.orientation.z;

        globalEstimator.InputOdom(last_vio_time_, vio_t, vio_q);

        gps_mutex_.lock();
        while (!gps_queue_.empty())
        {
            util::GpsPositionConstPtr GPS_msg = gps_queue_.front();
            double gps_t = GPS_msg->header.stamp.toSec();
            // 10ms sync tolerance
            if (gps_t >= last_vio_time_ - 0.03 && gps_t <= last_vio_time_ + 0.03)
            {
                double latitude = GPS_msg->latitude;
                double longitude = GPS_msg->longitude;
                double altitude = GPS_msg->height;

                double pos_accuracy = GPS_msg->gps_flag;
                if (pos_accuracy == 4)
                    globalEstimator.InputGPS(last_vio_time_, latitude, longitude, altitude, 1);

                gps_queue_.pop();
                break;
            }
            else if (gps_t < last_vio_time_ - 0.03)
                gps_queue_.pop();
            else if (gps_t > last_vio_time_ + 0.03)
                break;
        }
        gps_mutex_.unlock();

        Eigen::Vector3d global_t;
        Eigen::Quaterniond global_q;
        globalEstimator.GetGlobalOdom(global_t, global_q);

        if (vslam_init_flag_)
        {
            PublishLocation(global_t, global_q, pose_msg->header.stamp);
            PublishFusionTf(global_t, pose_msg->header.stamp);
        }
    }

    // Timer callback function, used to check self-test conditions and publish self-test results.
    void SelfCheckTimerCallback(const ros::TimerEvent &)
    {
        if (gps_state_ && vio_state_ && fusion_state_)
        {
            self_check_result_.data = true;
            // ROS_INFO("Camera self-check succeeded!");
        }
        else
        {
            self_check_result_.data = false;
            // ROS_WARN("Camera self-check failed!");
        }

        mower_msgs::VslamState msg;
        msg.is_vslam_ok = self_check_result_.data;
        // Release self-test results.
        self_check_pub_.publish(msg);

        // reset flag variable.
        gps_state_ = false;
        vio_state_ = false;
        fusion_state_ = false;
    }

private:
    void SaveGauss(double gauss_x, double gauss_y)
    {
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "gauss_x" << YAML::Value << gauss_x;
        out << YAML::Key << "gauss_y" << YAML::Value << gauss_y;
        out << YAML::EndMap;
        std::ofstream file_out("gauss_map.yaml");
        file_out << out.c_str();
        file_out.close();
        ROS_INFO("<---- Create map mode saved the gauss map! --->");
    }

    bool ReadGauss()
    {
        YAML::Node config = YAML::LoadFile("gauss_map.yaml");
        if (config.IsNull())
            return false;

        if (config["gauss_x"])
            read_gauss_x_ = config["gauss_x"].as<double>();
        if (config["gauss_y"])
            read_gauss_y_ = config["gauss_y"].as<double>();

        // Debug.
        // std::cout << "<--- Read gauss parameters: " << read_gauss_x_ << " " << read_gauss_y_ << " " << std::endl;

        if (read_gauss_x_ != 0 && read_gauss_y_ != 0)
            return true;

        return false;
    }

    void ComputeMatrix(double gauss_x, double gauss_y)
    {
        double delta_gauss_x = gauss_x - read_gauss_x_;
        double delta_gauss_y = gauss_y - read_gauss_y_;

        // Transform vector.
        Eigen::Vector3d translation(delta_gauss_x, -delta_gauss_y, 0.0);
        transformationMatrix_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        transformationMatrix_.block<3, 1>(0, 3) = translation;

        // Debug.
        // std::cout << "Matrix: " << std::endl
        //   << transformationMatrix_.matrix() << std::endl;
    }

    Eigen::Vector3d PointTranslation(Eigen::Vector4d matrix)
    {
        Eigen::Vector4d transformed_point = transformationMatrix_ * matrix;
        Eigen::Vector3d transformed_result = transformed_point.head(3);
        return transformed_result;
    }

    void PublishLocation(Eigen::Vector3d matrix, Eigen::Quaterniond global_q, ros::Time msg_time)
    {
        util::Vslam vslam_msg;
        vslam_msg.header.frame_id = car_frame_id_;
        vslam_msg.header.stamp = msg_time;

        if (!location_mode_)
        {
            vslam_msg.local_x = matrix.x();
            vslam_msg.local_y = matrix.y();
            vslam_msg.local_z = 0.; // TODO:

            tf2::Quaternion tf_quaternion(global_q.x(), global_q.y(), global_q.z(), global_q.w());
            tf2::Matrix3x3(tf_quaternion).getRPY(roll_, pitch_, azimuth_);

            vslam_msg.roll = 0.;
            vslam_msg.pitch = 0.;
            vslam_msg.yaw = azimuth_;
        }
        else
        {
            Eigen::Vector4d point(matrix.x(), matrix.y(), 0., 1.);
            Eigen::Vector3d transformed_point = PointTranslation(point);

            vslam_msg.local_x = transformed_point.x();
            vslam_msg.local_y = transformed_point.y();
            vslam_msg.local_z = 0.; // TODO:

            tf2::Quaternion tf_quaternion(global_q.x(), global_q.y(), global_q.z(), global_q.w());
            tf2::Matrix3x3(tf_quaternion).getRPY(roll_, pitch_, azimuth_);

            vslam_msg.roll = 0.;
            vslam_msg.pitch = 0.;
            vslam_msg.yaw = azimuth_;
        }

        fusion_pub_.publish(vslam_msg);

        fusion_state_ = true;

        // Debug code.
        /*
        if (save_csv_)
        {
            std::string file_path;
            if (!location_mode_)
                file_path = "./fusion_create.csv";
            else
                file_path = "./fusion_load.csv";
            std::ofstream foutC(file_path, ios::app);
            foutC.setf(ios::fixed, ios::floatfield);
            foutC << msg_time << " ";
            foutC.precision(5);
            foutC << matrix.x() << " "
                  << matrix.y() << " "
                  << matrix.z() << " "
                  << 0 << " "
                  << 0 << " "
                  << 0 << " "
                  << 0 << std::endl;
            foutC.close();
        }*/
    }

    // Publish the tf relationship function positioned after fusion.
    void PublishFusionTf(Eigen::Vector3d matrix, ros::Time msg_time)
    {
        tf::Quaternion tf_quaternion;
        tf_quaternion.setRPY(roll_, pitch_, azimuth_);
        tf::Vector3 tf_vec(matrix.x(), matrix.y(), matrix.z());

        static tf::TransformBroadcaster tf_broadcaster;
        tf_broadcaster.sendTransform(tf::StampedTransform(tf::Transform(tf_quaternion, tf_vec), msg_time, map_frame_id_, car_frame_id_));
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber gps_sub_, vslam_sub_, vslam_init_sub_, vslam_workingMode_sub_;
    ros::Publisher fusion_pub_, self_check_pub_;
    ros::Timer timer_;

    std::string node_name_, rtk_topic_, vslam_topic_,
        publish_topic_, vslam_state_, setting_file_;
    std::string map_frame_id_, car_frame_id_;
    std_msgs::Bool self_check_result_;
    bool gps_state_, vio_state_, fusion_state_, save_csv_;
    bool vslam_init_flag_, location_mode_;
    bool gps_ctr_flag_;
    double read_gauss_x_, read_gauss_y_;
    double pitch_, roll_, azimuth_;
    Eigen::Matrix4d transformationMatrix_;

private:
    GlobalOptimization globalEstimator;

    std::mutex gps_mutex_;
    std::queue<util::GpsPositionConstPtr> gps_queue_;

    double last_vio_time_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "FusionNode");

    FusionNode node;

    ros::spin();

    return 0;
}
