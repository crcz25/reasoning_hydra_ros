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
#pragma once

#include <config_utilities/factory.h>
#include <hydra/input/data_receiver.h>
#include <hydra/input/input_conversion.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>
#include <semantic_inference_msgs/FeatureImage.h>
#include <semantic_inference_msgs/FeatureVectorStamped.h>
#include <semantic_inference_msgs/FeatureVectorsStamped.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/Trigger.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

namespace hydra {

struct ImageSubscriber {
  ImageSubscriber();

  ImageSubscriber(const ros::NodeHandle& nh,
                  const std::string& camera_name,
                  const std::string& image_name = "image_raw",
                  uint32_t queue_size = 1);

  std::shared_ptr<image_transport::ImageTransport> transport;
  std::shared_ptr<image_transport::SubscriberFilter> sub;
};

class ImageReceiver : public DataReceiver {
 public:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image,
      sensor_msgs::Image,
      sensor_msgs::Image,
      semantic_inference_msgs::FeatureVectorStamped,
      semantic_inference_msgs::FeatureImage,
      semantic_inference_msgs::FeatureVectorsStamped>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  struct Config : DataReceiver::Config {
    std::string ns = "~";
    size_t queue_size = 100;
  };

  ImageReceiver(const Config& config, size_t sensor_id);

  virtual ~ImageReceiver();

 public:
  const Config config;

 protected:
  bool initImpl() override;

 private:
  bool toggleProcessingService(std_srvs::Trigger::Request& req,
                               std_srvs::Trigger::Response& res);
  void callback(
      const sensor_msgs::ImageConstPtr& color,
      const sensor_msgs::ImageConstPtr& depth,
      const sensor_msgs::ImageConstPtr& panoptic_ids,
      const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
      const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
      const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations);
  void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);
  void processLabels(
      const cv::Mat& panoptic_ids,
      const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
      ImageInputPacket::Ptr& packet,
      const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) const;

  message_filters::Subscriber<sensor_msgs::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> panoptic_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureVectorStamped>
      feature_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureImage> label_sub_;
  message_filters::Subscriber<semantic_inference_msgs::FeatureVectorsStamped>
      relations_sub_;
  ros::Subscriber camera_info_sub_;
  ros::NodeHandle nh_;
  std::unique_ptr<Synchronizer> synchronizer_;
  std::atomic<bool> processing_enabled_{true};
  ros::ServiceServer enable_processing_service_;

  inline static const auto registration_ =
      config::RegistrationWithConfig<DataReceiver,
                                     ImageReceiver,
                                     ImageReceiver::Config,
                                     size_t>("ImageReceiver");
  struct Maps {
    cv::Mat map1;
    cv::Mat map2;
    bool init = false;
  } maps_;
};

void declare_config(ImageReceiver::Config& config);

}  // namespace hydra
