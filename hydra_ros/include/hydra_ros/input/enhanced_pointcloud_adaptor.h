// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <glog/logging.h>
#include <hydra/input/input_module.h>
#include <hydra/input/sensor_input_packet.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <semantic_inference_msgs/FeatureImage.h>
#include <semantic_inference_msgs/FeatureVectorsStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <functional>
#include <opencv2/core.hpp>
#include <type_traits>

#include "hydra_ros/input/pointcloud_adaptor.h"

namespace hydra {

class EnhancedPointcloudAdaptor {
 public:
  explicit EnhancedPointcloudAdaptor(const sensor_msgs::PointCloud2& cloud);

  bool valid() const;

  cv::Vec3f position(const uint8_t* point_ptr) const;

 protected:
  std::function<double(const uint8_t*)> x_parser_;
  std::function<double(const uint8_t*)> y_parser_;
  std::function<double(const uint8_t*)> z_parser_;
  std::function<uint32_t(const uint8_t*)> label_parser_;
  std::function<cv::Vec3b(const uint8_t*)> color_parser_;
};

bool fillEnhancedPointcloudPacket(
    const PoseStatus lidar2cam,
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const sensor_msgs::PointCloud2ConstPtr& cloud,
    const cv::Mat& color,
    const cv::Mat& label_image,
    const cv::Mat& panoptic_ids,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations,
    const bool& undistort,
    EnhancedCloudInputPacket::Ptr& packet,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& debug_pointcloud);

Eigen::Vector3f projectPointcloudToImage(
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const PoseStatus& transform,
    const cv::Vec3f& point,
    const bool& undistort);

cv::Vec3b bilinearSampleVec3b(const cv::Mat& img, float u, float v);

uint16_t bilinearSampleUint16(const cv::Mat& img, float u, float v);

int32_t bilinearSampleInt32(const cv::Mat& img, float u, float v);

}  // namespace hydra
