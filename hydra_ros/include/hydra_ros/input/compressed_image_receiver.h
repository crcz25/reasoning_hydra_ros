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
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <std_srvs/Trigger.h>

#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

namespace hydra {

enum compressionFormat { UNDEFINED = -1, INV_DEPTH };

struct ConfigHeader {
  // compression format
  compressionFormat format;
  // quantization parameters (used in depth image compression)
  float depthParam[2];
};

class CompressedImageReceiver : public DataReceiver {
 public:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CompressedImage,
      sensor_msgs::CompressedImage,
      sensor_msgs::Image,
      semantic_inference_msgs::FeatureVectorStamped,
      semantic_inference_msgs::FeatureImage,
      semantic_inference_msgs::FeatureVectorsStamped>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  struct Config : DataReceiver::Config {
    std::string ns = "~";
    size_t queue_size = 100;
  };

  CompressedImageReceiver(const Config& config, size_t sensor_id);

  virtual ~CompressedImageReceiver();

 public:
  const Config config;

 protected:
  bool initImpl() override;

 private:
  bool toggleProcessingService(std_srvs::Trigger::Request& req,
                               std_srvs::Trigger::Response& res);
  void callback(
      const sensor_msgs::CompressedImageConstPtr& color,
      const sensor_msgs::CompressedImageConstPtr& depth,
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

  sensor_msgs::Image::Ptr decodeCompressedDepthImage(
      const sensor_msgs::CompressedImage& message) const;
  message_filters::Subscriber<sensor_msgs::CompressedImage> color_sub_;
  message_filters::Subscriber<sensor_msgs::CompressedImage> depth_sub_;
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
                                     CompressedImageReceiver,
                                     CompressedImageReceiver::Config,
                                     size_t>("CompressedImageReceiver");
  struct Maps {
    cv::Mat map1;
    cv::Mat map2;
    bool init = false;
  } maps_;
};

void declare_config(CompressedImageReceiver::Config& config);

}  // namespace hydra
