// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/input/pointcloud_image_receiver.h"

namespace hydra {

void declare_config(PointcloudImageReceiver::Config& config) {
  using namespace config;
  name("PointcloudImageReceiver::Config");
  base<DataReceiver::Config>(config);
  field(config.ns, "ns");
  field(config.queue_size, "queue_size");
  field(config.depth_debug, "depth_debug");
  field(config.tf_wait_duration_s, "tf_wait_duration_s", "s");
  field(config.tf_buffer_size_s, "tf_buffer_size_s");
  field(config.tf_max_tries, "tf_max_tries");
  field(config.tf_verbosity, "tf_verbosity");
  field(config.undistort, "undistort");
  field(config.slop_time_s, "slop_time_s", "s");
}

PointcloudImageReceiver::PointcloudImageReceiver(const Config& config, size_t sensor_id)
    : DataReceiver(config, sensor_id), config(config), nh_(config.ns) {
  buffer_.reset(new tf2_ros::Buffer(ros::Duration(config.tf_buffer_size_s)));
  tf_listener_.reset(new tf2_ros::TransformListener(*buffer_));
}

PointcloudImageReceiver::~PointcloudImageReceiver() {}

bool PointcloudImageReceiver::initImpl() {
  // TODO(nathan) subscribe to image subsets

  pcl_sub_.subscribe(nh_, "pointcloud", config.queue_size);
  color_sub_.subscribe(nh_, "rgb/image_raw", config.queue_size);
  camera_info_sub_.subscribe(nh_, "rgb/camera_info", config.queue_size);
  panoptic_sub_.subscribe(nh_, "panoptic/image_raw", config.queue_size);
  feature_sub_.subscribe(nh_, "image_feature", config.queue_size);
  label_sub_.subscribe(nh_, "semantic", config.queue_size);
  relations_sub_.subscribe(nh_, "relations", config.queue_size);
  synchronizer_.reset(new Synchronizer(SyncPolicy(config.queue_size),
                                       pcl_sub_,
                                       color_sub_,
                                       camera_info_sub_,
                                       panoptic_sub_,
                                       feature_sub_,
                                       label_sub_,
                                       relations_sub_));
  synchronizer_->setMaxIntervalDuration(ros::Duration(config.slop_time_s));
  synchronizer_->registerCallback(boost::bind(
      &PointcloudImageReceiver::callback, this, _1, _2, _3, _4, _5, _6, _7));
  // Print full topic names for debugging.
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", pcl_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", color_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", camera_info_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", panoptic_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", feature_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", label_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Subscribed to: %s", relations_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[pointcloud_image_receiver] Queue size: %zu", config.queue_size);
  ROS_INFO("[pointcloud_image_receiver] Slop time: %f", config.slop_time_s);
  ROS_INFO("[pointcloud_image_receiver] TF buffer size: %f", config.tf_buffer_size_s);
  ROS_INFO("[pointcloud_image_receiver] TF wait duration: %f", config.tf_wait_duration_s);
  ROS_INFO("[pointcloud_image_receiver] TF max tries: %zu", config.tf_max_tries);

  debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("debug_pointcloud", 1, true);
  return true;
}

PoseStatus PointcloudImageReceiver::getTransform(const std::string& target,
                                                 const std::string& source,
                                                 const ros::Time& time) {
  // negative or 0 for tf_max_tries means we spin forever if the transform isn't present
  // const std::optional<size_t> max_tries =
  //     config.tf_max_tries > 0 ? std::optional<size_t>(config.tf_max_tries)
  //                             : std::nullopt;

  const auto pose_status = lookupTransformNoBLock(
      *buffer_, time, target, source, config.tf_wait_duration_s, config.tf_verbosity);
  return pose_status;
}

void PointcloudImageReceiver::callback(
    const sensor_msgs::PointCloud2ConstPtr& cloud,
    const sensor_msgs::ImageConstPtr& color,
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const sensor_msgs::ImageConstPtr& panoptic_ids,
    const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) {
  // Log info color and cloud timestamps and difference
  const auto difference =
      color->header.stamp.toNSec() > cloud->header.stamp.toNSec()
          ? color->header.stamp.toNSec() - cloud->header.stamp.toNSec()
          : cloud->header.stamp.toNSec() - color->header.stamp.toNSec();
  const auto labels_image = boost::make_shared<sensor_msgs::Image>(labels->image);
  if (labels_image &&
      (labels_image->width != color->width || labels_image->height != color->height)) {
    LOG(ERROR) << "Label dimensions do not match color dimensions!";
    return;
  }
  if (!checkInputTimestamp(color->header.stamp.toNSec())) {
    return;
  }

  auto packet = std::make_shared<EnhancedCloudInputPacket>(color->header.stamp.toNSec(),
                                                           sensor_id_);

  const auto lidar2cam =
      getTransform(color->header.frame_id, cloud->header.frame_id, color->header.stamp);

  if (!lidar2cam) {
    LOG(ERROR) << "Failed to find transform from " << color->header.frame_id << " to "
               << cloud->header.frame_id;
    return;
  }

  cv::Mat label_image;
  if (!hydra::conversions::colorToLabels(
          label_image, cv_bridge::toCvShare(labels_image)->image.clone())) {
    return;
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_pointcloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  fillEnhancedPointcloudPacket(lidar2cam,
                               camera_info,
                               cloud,
                               cv_bridge::toCvShare(color)->image,
                               cv_bridge::toCvShare(labels_image)->image,
                               cv_bridge::toCvShare(panoptic_ids)->image,
                               labels,
                               relations,
                               config.undistort,
                               packet,
                               debug_pointcloud);

  if (debug_pointcloud->size() > 0) {
    sensor_msgs::PointCloud2 debug_msg;
    pcl::toROSMsg(*debug_pointcloud, debug_msg);
    debug_msg.header = cloud->header;
    debug_pub_.publish(debug_msg);
  }

  if (!features->feature.data.empty()) {
    // Map vector of floats to Eigen vector.
    packet->image_feature = Eigen::Map<const Eigen::VectorXf>(
        features->feature.data.data(), features->feature.data.size());
  }
  packet->cam_T_lidar = lidar2cam.toIsometry();
  queue.push(packet);
}
}  // namespace hydra
