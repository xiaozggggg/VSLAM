#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <ros/time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <tf/transform_broadcaster.h>
#include <image_transport/image_transport.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <std_msgs/String.h>

// ORB-SLAM3-specific libraries
#include "System.h"
#include "ImuTypes.h"

// Mr.Chen: my new msg
#include "util/GpsPosition.h"
#include "util/Vslam.h"
#include "mower_msgs/VslamState.h"

// Mr.Chen:新增加tf转换测试(2023.11.20)
#include <tf/transform_broadcaster.h>

extern ORB_SLAM3::System *pSLAM;
extern ORB_SLAM3::System::eSensor sensor_type;

extern std::string world_frame_id, cam_frame_id, imu_frame_id;

// Mr.Chen:add_new.
// extern std::string map_frame_id,car_frame_id, vslam_frame_id;

extern ros::Publisher pose_pub, odom_pub, kf_markers_pub, pose_fusion;
extern ros::Publisher tracked_mappoints_pub, all_mappoints_pub;
extern image_transport::Publisher tracking_img_pub;

// yolo
extern ros::Publisher yolo_pub;

void setup_services(ros::NodeHandle &, std::string);
void setup_publishers(ros::NodeHandle &, image_transport::ImageTransport &, std::string);
void publish_topics(ros::Time, Eigen::Vector3f = Eigen::Vector3f::Zero());

void publish_camera_pose(Sophus::SE3f, ros::Time);

// -----------------------------------------------
// Mr.Chen: publish new position message
// 该话题是相机坐标系到车体坐标系变换后的发布函数(vslam)
void publish_cam2_car(Sophus::SE3f, ros::Time);
// tf关系发布
void publish_camtf_car(Sophus::SE3f, string, string, ros::Time);
// -----------------------------------------------

void publish_tracking_img(cv::Mat, ros::Time);
void publish_tracked_points(std::vector<ORB_SLAM3::MapPoint *>, ros::Time);
void publish_all_points(std::vector<ORB_SLAM3::MapPoint *>, ros::Time);
void publish_tf_transform(Sophus::SE3f, string, string, ros::Time);
void publish_body_odom(Sophus::SE3f, Eigen::Vector3f, Eigen::Vector3f, ros::Time);
void publish_kf_markers(std::vector<Sophus::SE3f>, ros::Time);
// yolo
void publish_yolo_data(ros::Time msg_time);

cv::Mat SE3f_to_cvMat(Sophus::SE3f);
tf::Transform SE3f_to_tfTransform(Sophus::SE3f);
sensor_msgs::PointCloud2 mappoint_to_pointcloud(std::vector<ORB_SLAM3::MapPoint *>, ros::Time);

