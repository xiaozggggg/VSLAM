#ifndef GLOBAL_FUSION_H
#define GLOBAL_FUSION_H

#include <iostream>
#include <thread>
#include <map>
#include <vector>
#include <mutex>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include "LocalCartesian.hpp"
#include "Factors.h"

// using namespace std;
// using namespace ceres;

class GlobalOptimization
{
public:
    GlobalOptimization();
    ~GlobalOptimization();

    void InputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy);
    void InputOdom(double t,Eigen::Vector3d OdomP, Eigen::Quaterniond OdomQ);
    void GetGlobalOdom(Eigen::Vector3d &odomP, Eigen::Quaterniond &odomQ);

private:
    void GPS2XYZ(double latitude, double longitude, double altitude, double *xyz);
    void optimize();

    bool mbInitGPS;
    bool mbGPSCtrl;
    bool mbThreadCtrl;

    std::map<double, std::vector<double>> localPoseMap;
	std::map<double, std::vector<double>> globalPoseMap;
    std::map<double, std::vector<double>> GPSPositionMap;

    GeographicLib::LocalCartesian geoConverter;
    Eigen::Matrix4d WGPS_T_WVIO;
    Eigen::Vector3d lastP;
	Eigen::Quaterniond lastQ;

    std::mutex mtxOdm;
    std::thread optThread;
};

#endif // GLOBAL_FUSION_H