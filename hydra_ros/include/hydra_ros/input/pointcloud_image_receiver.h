// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once
#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <hydra/common/global_info.h>
#include <hydra/input/data_receiver.h>
#include <hydra/input/input_conversion.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <semantic_inference_msgs/FeatureImage.h>
#include <semantic_inference_msgs/FeatureVectorStamped.h>
#include <semantic_inference_msgs/FeatureVectorsStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <memory>
#include <string>

#include "hydra_ros/input/enhanced_pointcloud_adaptor.h"
#include "hydra_ros/utils/lookup_tf.h"

namespace hydra {

class PointcloudImageReceiver : public DataReceiver {
 public:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2,
      sensor_msgs::Image,
      sensor_msgs::CameraInfo,
      sensor_msgs::Image,
      semantic_inference_msgs::FeatureVectorStamped,
      semantic_inference_msgs::FeatureImage,
      semantic_inference_msgs::FeatureVectorsStamped>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  struct Config : DataReceiver::Config {
    std::string ns = "~";
    size_t queue_size = 100;
    bool depth_debug = true;
    //! Amount of time to wait between tf lookup attempts
    double tf_wait_duration_s = 0.1;
    //! Buffer size in second for tf
    double tf_buffer_size_s = 30.0;
    //! Number of lookup attempts before giving up
    size_t tf_max_tries = 5;
    //! Logging verbosity of tf lookup process
    int tf_verbosity = 3;
    //! Undistort flag
    bool undistort = false;
    //! Time to wait before processing the next input
    double slop_time_s = 0.1;
  } const config;

  PointcloudImageReceiver(const Config& config, size_t sensor_id);

  virtual ~PointcloudImageReceiver();

 protected:
  bool initImpl() override;

 private:
  void callback(
      const sensor_msgs::PointCloud2ConstPtr& cloud,
      const sensor_msgs::ImageConstPtr& color,
      const sensor_msgs::CameraInfoConstPtr& camera_info,
      const sensor_msgs::ImageConstPtr& panoptic_ids,
      const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
      const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
      const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations);

  PoseStatus getTransform(const std::string& target,
                          const std::string& source,
                          const ros::Time& time);

  message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub_;
  message_filters::Subscriber<sensor_msgs::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::Image> panoptic_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureVectorStamped>
      feature_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureImage> label_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureVectorsStamped>
      relations_sub_;
  std::unique_ptr<Synchronizer> synchronizer_;
  ros::NodeHandle nh_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  ros::Publisher debug_pub_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<DataReceiver,
                                     PointcloudImageReceiver,
                                     PointcloudImageReceiver::Config,
                                     size_t>("PointcloudImageReceiver");
};

void declare_config(PointcloudImageReceiver::Config& config);

}  // namespace hydra
