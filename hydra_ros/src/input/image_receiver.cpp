// Portions of the following code and their modifications are originally from
// https://github.com/MIT-SPARK/Hydra/tree/main and are licensed under the following
// license:
/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */

// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/input/image_receiver.h"

#include <config_utilities/config.h>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <ros/console.h>

namespace hydra {

using image_transport::ImageTransport;
using image_transport::SubscriberFilter;

image_transport::TransportHints getHintsWithNamespace(const ros::NodeHandle& nh,
                                                      const std::string& ns) {
  return image_transport::TransportHints(
      "raw", ros::TransportHints(), ros::NodeHandle(nh, ns));
}

void declare_config(ImageReceiver::Config& config) {
  using namespace config;
  name("ImageReceiver::Config");
  base<DataReceiver::Config>(config);
  field(config.ns, "ns");
  field(config.queue_size, "queue_size");
}

ImageSubscriber::ImageSubscriber() {}

ImageSubscriber::ImageSubscriber(const ros::NodeHandle& nh,
                                 const std::string& camera_name,
                                 const std::string& image_name,
                                 uint32_t queue_size)
    : transport(std::make_shared<ImageTransport>(ros::NodeHandle(nh, camera_name))),
      sub(std::make_shared<SubscriberFilter>(
          *transport, image_name, queue_size, getHintsWithNamespace(nh, camera_name))) {
}

ImageReceiver::ImageReceiver(const Config& config, size_t sensor_id)
    : DataReceiver(config, sensor_id), config(config), nh_(config.ns) {}

bool ImageReceiver::initImpl() {
  // TODO(nathan) subscribe to image subsets
  // color_sub_ = ImageSubscriber(nh_, "rgb");
  // depth_sub_ = ImageSubscriber(nh_, "depth_registered", "image_rect");

  enable_processing_service_ = nh_.advertiseService(
      "toggle_image_processing", &ImageReceiver::toggleProcessingService, this);

  camera_info_sub_ = nh_.subscribe(
      "rgb/camera_info", config.queue_size, &ImageReceiver::callbackCameraInfo, this);
  color_sub_.subscribe(nh_, "rgb/image_raw", config.queue_size);
  depth_sub_.subscribe(nh_, "depth_registered/image_rect", config.queue_size);
  panoptic_sub_.subscribe(nh_, "panoptic/image_raw", config.queue_size);
  feature_sub_.subscribe(nh_, "image_feature", config.queue_size);
  label_sub_.subscribe(nh_, "semantic", config.queue_size);
  relations_sub_.subscribe(nh_, "relations", config.queue_size);
  synchronizer_.reset(new Synchronizer(SyncPolicy(config.queue_size),
                                       color_sub_,
                                       depth_sub_,
                                       panoptic_sub_,
                                       feature_sub_,
                                       label_sub_,
                                       relations_sub_));
  synchronizer_->registerCallback(
      boost::bind(&ImageReceiver::callback, this, _1, _2, _3, _4, _5, _6));
  // Print full topic names for debugging.
  ROS_INFO("[image_receiver] Subscribed to: %s", color_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[image_receiver] Subscribed to: %s", depth_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[image_receiver] Subscribed to: %s", panoptic_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[image_receiver] Subscribed to: %s", feature_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[image_receiver] Subscribed to: %s", label_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[image_receiver] Subscribed to: %s", relations_sub_.getSubscriber().getTopic().c_str());

  return true;
}

ImageReceiver::~ImageReceiver() {}

bool ImageReceiver::toggleProcessingService(std_srvs::Trigger::Request& req,
                                            std_srvs::Trigger::Response& res) {
  processing_enabled_ = !processing_enabled_;
  res.success = true;
  res.message = processing_enabled_ ? "Processing enabled" : "Processing disabled";
  ROS_INFO("[image_receiver] Image processing %s",
           processing_enabled_ ? "enabled" : "disabled");
  return true;
}

std::string showImageDim(const sensor_msgs::Image::ConstPtr& image) {
  std::stringstream ss;
  ss << "[" << image->width << ", " << image->height << "]";
  return ss.str();
}

void ImageReceiver::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg) {
  if (msg->distortion_model == "none") {
    LOG(WARNING) << "Camera info has no distortion model. Skipping initialization.";
    maps_.init = true;  // No distortion, so we can consider it initialized.
    camera_info_sub_.shutdown();
    return;
  }
  cv::Mat camera_matrix(3, 3, CV_64F, const_cast<double*>(msg->K.data()));
  cv::Mat dist_coeffs = cv::Mat(msg->D).clone();
  cv::Size image_size(msg->width, msg->height);
  cv::initUndistortRectifyMap(camera_matrix,
                              dist_coeffs,
                              cv::Mat(),
                              camera_matrix,  // or newCameraMatrix
                              image_size,
                              CV_16SC2,
                              maps_.map1,
                              maps_.map2);
  maps_.init = true;
  ROS_INFO("[image_receiver] Initialized camera maps for undistortion and rectification.");
  camera_info_sub_.shutdown();
}

void ImageReceiver::processLabels(
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

void ImageReceiver::callback(
    const sensor_msgs::ImageConstPtr& color,
    const sensor_msgs::ImageConstPtr& depth,
    const sensor_msgs::ImageConstPtr& panoptic_ids,
    const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) {
  if (!maps_.init) {
    LOG(ERROR) << "Camera maps are not initialized. Cannot process images.";
    return;
  }
  if (!processing_enabled_) {
    ROS_WARN_THROTTLE(5.0, "[image_receiver] Image processing paused via service toggle.");
    return;
  }

  if (color && (color->width != depth->width || color->height != depth->height)) {
    LOG(ERROR) << "color dimensions do not match depth dimensions: "
               << showImageDim(color) << " != " << showImageDim(depth);
    return;
  }
  const auto labels_image = boost::make_shared<sensor_msgs::Image>(labels->image);
  if (labels_image &&
      (labels_image->width != depth->width || labels_image->height != depth->height)) {
    LOG(ERROR) << "label dimensions do not match depth dimensions: "
               << showImageDim(labels_image) << " != " << showImageDim(depth);
    return;
  }

  if (!checkInputTimestamp(depth->header.stamp.toNSec())) {
    return;
  }

  auto packet =
      std::make_shared<ImageInputPacket>(color->header.stamp.toNSec(), sensor_id_);
  try {
    const auto cv_depth = cv_bridge::toCvShare(depth);
    packet->depth = cv_depth->image.clone();
    // Convert depth to meters if it is in millimeters.
    if (cv_depth->image.type() == CV_16UC1) {
      packet->depth.convertTo(packet->depth, CV_32FC1, 0.001);
    } else if (cv_depth->image.type() != CV_32FC1) {
      LOG(ERROR) << "Unsupported depth image type: " << cv_depth->image.type();
      return;
    }
    if (color && color->encoding == sensor_msgs::image_encodings::RGB8) {
      auto cv_color = cv_bridge::toCvShare(color);
      packet->color = cv_color->image.clone();
    } else if (color) {
      auto cv_color = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::RGB8);
      packet->color = cv_color->image;
    }
    if (!maps_.map1.empty() && !maps_.map2.empty()) {
      // Apply the camera maps for undistortion and rectification.
      cv::remap(packet->depth, packet->depth, maps_.map1, maps_.map2, cv::INTER_LINEAR);
      cv::remap(packet->color, packet->color, maps_.map1, maps_.map2, cv::INTER_LINEAR);
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
}
}  // namespace hydra
