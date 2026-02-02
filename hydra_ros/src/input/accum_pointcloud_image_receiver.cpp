// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/input/accum_pointcloud_image_receiver.h"

namespace hydra {

void declare_config(AccumPointcloudImageReceiver::Config& config) {
  using namespace config;
  name("AccumPointcloudImageReceiver::Config");
  base<DataReceiver::Config>(config);
  field(config.ns, "ns");
  field(config.queue_size, "queue_size");
  field(config.voxel_size, "voxel_size");
  field(config.world_frame, "world_frame");
  field(config.cam_frame, "cam_frame");
  field(config.depth_debug, "depth_debug");
  field(config.tf_wait_duration_s, "tf_wait_duration_s");
  field(config.tf_buffer_size_s, "tf_buffer_size_s");
  field(config.tf_max_tries, "tf_max_tries");
  field(config.tf_verbosity, "tf_verbosity");
}

AccumPointcloudImageReceiver::AccumPointcloudImageReceiver(const Config& config,
                                                           size_t sensor_id)
    : DataReceiver(config, sensor_id), config(config), nh_(config.ns) {
  buffer_.reset(new tf2_ros::Buffer(ros::Duration(config.tf_buffer_size_s)));
  tf_listener_.reset(new tf2_ros::TransformListener(*buffer_));
  accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());
}

AccumPointcloudImageReceiver::~AccumPointcloudImageReceiver() {}

bool AccumPointcloudImageReceiver::initImpl() {
  // TODO(nathan) subscribe to image subsets

  depth_debug_pub_ = nh_.advertise<sensor_msgs::Image>("depth_debug", 1, true);
  cloud_sub_ = nh_.subscribe("pointcloud",
                             config.queue_size,
                             &AccumPointcloudImageReceiver::pointcloudCallback,
                             this);

  color_sub_.subscribe(nh_, "rgb/image_raw", config.queue_size);
  camera_info_sub_.subscribe(nh_, "rgb/camera_info", config.queue_size);
  panoptic_sub_.subscribe(nh_, "panoptic/image_raw", config.queue_size);
  feature_sub_.subscribe(nh_, "image_feature", config.queue_size);
  label_sub_.subscribe(nh_, "semantic", config.queue_size);
  relations_sub_.subscribe(nh_, "relations", config.queue_size);
  synchronizer_.reset(new Synchronizer(SyncPolicy(config.queue_size),
                                       color_sub_,
                                       camera_info_sub_,
                                       panoptic_sub_,
                                       feature_sub_,
                                       label_sub_,
                                       relations_sub_));
  synchronizer_->registerCallback(boost::bind(
      &AccumPointcloudImageReceiver::callback, this, _1, _2, _3, _4, _5, _6));
  // Print full topic names for debugging.
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", color_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", camera_info_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", panoptic_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", feature_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", label_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[accum_pointcloud_image_receiver] Subscribed to: %s", relations_sub_.getSubscriber().getTopic().c_str());

  return true;
}

Eigen::Matrix4f AccumPointcloudImageReceiver::transformToMatrix(
    const geometry_msgs::TransformStamped& tf_msg) const {
  Eigen::Translation3f translation(tf_msg.transform.translation.x,
                                   tf_msg.transform.translation.y,
                                   tf_msg.transform.translation.z);
  Eigen::Quaternionf rotation(tf_msg.transform.rotation.w,
                              tf_msg.transform.rotation.x,
                              tf_msg.transform.rotation.y,
                              tf_msg.transform.rotation.z);
  return (translation * rotation).matrix();
}

PoseStatus AccumPointcloudImageReceiver::getTransform(const std::string& target,
                                                      const std::string& source,
                                                      const ros::Time& time) {
  // negative or 0 for tf_max_tries means we spin forever if the transform isn't present
  const std::optional<size_t> max_tries =
      config.tf_max_tries > 0 ? std::optional<size_t>(config.tf_max_tries)
                              : std::nullopt;

  const auto pose_status = lookupTransform(*buffer_,
                                           time,
                                           target,
                                           source,
                                           max_tries,
                                           config.tf_wait_duration_s,
                                           config.tf_verbosity);
  return pose_status;
}

void AccumPointcloudImageReceiver::processLabels(
    const cv::Mat& panoptic_ids,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    ImageInputPacket::Ptr& packet,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) const {
  // Initialize the features mask
  packet->features_mask = cv::Mat::zeros(panoptic_ids.size(), CV_16UC1);
  panoptic_ids.convertTo(packet->features_mask.value(), CV_16UC1);

  // Reserve space for the semantic_features map
  packet->semantic_features = std::unordered_map<uint16_t, Eigen::VectorXf>();
  packet->semantic_features.value().reserve(labels->mask_ids.size());
  for (size_t i = 0; i < labels->mask_ids.size(); ++i) {
    const auto& mask_id = labels->mask_ids[i];
    const auto& feature_data = labels->features[i].data;
    packet->semantic_features.value()[mask_id] =
        Eigen::Map<const Eigen::VectorXf>(feature_data.data(), feature_data.size());
  }

  if (!relations->feature.empty()) {
    packet->relations = PairHashMap();
  }

  for (size_t i = 0; i < static_cast<size_t>(relations->choice.size() / 2); ++i) {
    const auto& relation =
        relations->feature.size() > 1 ? relations->feature[i] : relations->feature[0];
    try {
      packet->relations.value()[std::make_pair(
          static_cast<uint16_t>(relations->choice[2 * i]),
          static_cast<uint16_t>(relations->choice[2 * i + 1]))] =
          Eigen::Map<const Eigen::MatrixXf>(
              relation.data.data(), relation.rows, relation.cols);
    } catch (const std::exception& e) {
      LOG(ERROR) << "unable to read relations from ros: " << e.what();
    }
  }
}

void AccumPointcloudImageReceiver::pointcloudCallback(
    const sensor_msgs::PointCloud2ConstPtr& msg) {
  const auto pose_status =
      getTransform(config.world_frame, msg->header.frame_id, msg->header.stamp);
  if (!pose_status) {
    LOG(ERROR) << "Failed to find transform from " << config.world_frame << " to "
               << msg->header.frame_id;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *input_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::transformPointCloud(
      *input_cloud, *transformed, pose_status.toMatrix().cast<float>());

  std::lock_guard<std::mutex> lock(mutex_);
  *accumulated_cloud_ += *transformed;
  pcl::VoxelGrid<pcl::PointXYZ> voxel;
  voxel.setInputCloud(accumulated_cloud_);
  voxel.setLeafSize(config.voxel_size, config.voxel_size, config.voxel_size);
  voxel.filter(*accumulated_cloud_);
}

void AccumPointcloudImageReceiver::callback(
    const sensor_msgs::ImageConstPtr& color,
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const sensor_msgs::ImageConstPtr& panoptic_ids,
    const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) {
  const auto labels_image = boost::make_shared<sensor_msgs::Image>(labels->image);

  if (!checkInputTimestamp(color->header.stamp.toNSec())) {
    return;
  }

  auto packet =
      std::make_shared<ImageInputPacket>(color->header.stamp.toNSec(), sensor_id_);

  const auto pose_status =
      getTransform(config.cam_frame, config.world_frame, color->header.stamp);

  if (!pose_status) {
    LOG(ERROR) << "Failed to find transform from " << config.cam_frame << " to "
               << config.world_frame;
    return;
  }

  if (accumulated_cloud_->empty()) {
    LOG(ERROR) << "Accumulated point cloud is empty.";
    return;
  }

  try {
    packet->depth = projectPointcloudToImage(camera_info, pose_status);
    if (color && color->encoding == sensor_msgs::image_encodings::RGB8) {
      auto cv_color = cv_bridge::toCvShare(color);
      packet->color = cv_color->image.clone();
    } else if (color) {
      auto cv_color = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::RGB8);
      packet->color = cv_color->image;
    }

    if (labels_image) {
      auto cv_labels = cv_bridge::toCvShare(labels_image);
      packet->labels = cv_labels->image.clone();
    }
  } catch (const cv_bridge::Exception& e) {
    LOG(ERROR) << "unable to read images from ros: " << e.what();
  }

  cv::Mat label_image;
  if (!hydra::conversions::colorToLabels(label_image, packet->labels)) {
    return;
  }
  if (!labels->features.empty() || !relations->feature.empty()) {
    auto cv_panoptic = cv_bridge::toCvShare(panoptic_ids);
    processLabels(cv_panoptic->image, labels, packet, relations);
  }

  if (!features->feature.data.empty()) {
    // Map vector of floats to Eigen vector.
    packet->image_feature = Eigen::Map<const Eigen::VectorXf>(
        features->feature.data.data(), features->feature.data.size());
  }
  queue.push(packet);

  if (config.depth_debug) {
    sensor_msgs::Image depth_debug;
    cv_bridge::CvImage(
        color->header, sensor_msgs::image_encodings::TYPE_32FC1, packet->depth)
        .toImageMsg(depth_debug);
    depth_debug_pub_.publish(depth_debug);
  }
}

cv::Mat AccumPointcloudImageReceiver::projectPointcloudToImage(
    const sensor_msgs::CameraInfoConstPtr& camera_info, const PoseStatus& transform) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Create an empty depth image initialized to zero
  cv::Mat depth_image = cv::Mat::zeros(camera_info->height, camera_info->width, CV_32F);

  // Camera intrinsic parameters
  float fx = camera_info->K[0];  // Focal length in x
  float fy = camera_info->K[4];  // Focal length in y
  float cx = camera_info->K[2];  // Optical center x
  float cy = camera_info->K[5];  // Optical center y

  // Distortion coefficients (equidistant model)
  double k1 = camera_info->D[0];  // k1
  double k2 = camera_info->D[1];  // k2
  double k3 = camera_info->D[2];  // k3
  double k4 = camera_info->D[3];  // k4

  // Iterate through the accumulated point cloud and project valid points
  for (const auto& point : *accumulated_cloud_) {
    // Transform the point from world frame to camera frame
    Eigen::Vector4f pt_world(point.x, point.y, point.z, 1.0);
    Eigen::Vector4f pt_camera = transform.toMatrix().cast<float>() * pt_world;

    // Extract camera frame coordinates
    float x_cam = pt_camera(0);
    float y_cam = pt_camera(1);
    float z_cam = pt_camera(2);

    // Skip points that are behind the camera (z <= 0)
    if (z_cam <= 0) {
      continue;
    }

    // Project the point to the image plane (camera intrinsics)
    float u = fx * (x_cam / z_cam) + cx;
    float v = fy * (y_cam / z_cam) + cy;

    // Check if the projected point lies within the image boundaries
    if (u >= 0 && u < camera_info->width && v >= 0 && v < camera_info->height) {
      // Compute the radial distance from the principal point
      float r2 =
          (x_cam * x_cam + y_cam * y_cam) / (z_cam * z_cam);  // radial distance squared

      // Apply the equidistant distortion model
      float distortion_factor =
          (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2);

      // Apply the distortion to the u, v coordinates
      float u_d = u * distortion_factor;
      float v_d = v * distortion_factor;

      // Ensure the projected pixel is within valid range
      if (u_d >= 0 && u_d < camera_info->width && v_d >= 0 &&
          v_d < camera_info->height) {
        // Set the depth image pixel value (in meters)
        depth_image.at<float>(v_d, u_d) = z_cam;
      }
    }
  }

  return depth_image;  // Return the depth image
}
}  // namespace hydra
