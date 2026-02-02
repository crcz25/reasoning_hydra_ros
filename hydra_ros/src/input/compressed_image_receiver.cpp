// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/input/compressed_image_receiver.h"

#include <config_utilities/config.h>
#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <ros/console.h>

namespace hydra {

void declare_config(CompressedImageReceiver::Config& config) {
  using namespace config;
  name("CompressedImageReceiver::Config");
  base<DataReceiver::Config>(config);
  field(config.ns, "ns");
  field(config.queue_size, "queue_size");
}

CompressedImageReceiver::CompressedImageReceiver(const Config& config, size_t sensor_id)
    : DataReceiver(config, sensor_id), config(config), nh_(config.ns) {}

bool CompressedImageReceiver::initImpl() {
  // TODO(nathan) subscribe to image subsets
  // color_sub_ = ImageSubscriber(nh_, "rgb");
  // depth_sub_ = ImageSubscriber(nh_, "depth_registered", "image_rect");

  enable_processing_service_ =
      nh_.advertiseService("toggle_image_processing",
                           &CompressedImageReceiver::toggleProcessingService,
                           this);

  camera_info_sub_ = nh_.subscribe("rgb/camera_info",
                                   config.queue_size,
                                   &CompressedImageReceiver::callbackCameraInfo,
                                   this);
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
      boost::bind(&CompressedImageReceiver::callback, this, _1, _2, _3, _4, _5, _6));
  // Print full topic names for debugging.
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", color_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", depth_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", panoptic_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", feature_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", label_sub_.getSubscriber().getTopic().c_str());
  ROS_INFO("[compressed_image_receiver] Subscribed to: %s", relations_sub_.getSubscriber().getTopic().c_str());

  return true;
}

CompressedImageReceiver::~CompressedImageReceiver() {}

bool CompressedImageReceiver::toggleProcessingService(
    std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
  processing_enabled_ = !processing_enabled_;
  res.success = true;
  res.message = processing_enabled_ ? "Processing enabled" : "Processing disabled";
  ROS_INFO("[compressed_image_receiver] Image processing %s",
           processing_enabled_ ? "enabled" : "disabled");
  return true;
}

void CompressedImageReceiver::callbackCameraInfo(
    const sensor_msgs::CameraInfoConstPtr& msg) {
  cv::Mat camera_matrix = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();
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
  ROS_INFO("[compressed_image_receiver] Initialized camera maps for undistortion and rectification.");
  camera_info_sub_.shutdown();
}

void CompressedImageReceiver::processLabels(
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

sensor_msgs::Image::Ptr CompressedImageReceiver::decodeCompressedDepthImage(
    const sensor_msgs::CompressedImage& message) const {
  cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

  // Copy message header
  cv_ptr->header = message.header;

  // Assign image encoding
  const size_t split_pos = message.format.find(';');
  const std::string image_encoding = message.format.substr(0, split_pos);
  std::string compression_format;
  // Older version of compressed_depth_image_transport supports only png.
  if (split_pos == std::string::npos) {
    compression_format = "png";
  } else {
    std::string format = message.format.substr(split_pos);
    if (format.find("compressedDepth png") != std::string::npos) {
      compression_format = "png";
    } else if (format.find("compressedDepth rvl") != std::string::npos) {
      compression_format = "rvl";
    } else if (format.find("compressedDepth") != std::string::npos &&
               format.find("compressedDepth ") == std::string::npos) {
      compression_format = "png";
    } else {
      ROS_ERROR("[compressed_image_receiver] Unsupported image format: %s", message.format.c_str());
      return sensor_msgs::Image::Ptr();
    }
  }

  cv_ptr->encoding = image_encoding;

  // Decode message data
  if (message.data.size() > sizeof(ConfigHeader)) {
    // Read compression type from stream
    ConfigHeader compressionConfig;
    memcpy(&compressionConfig, &message.data[0], sizeof(compressionConfig));

    // Get compressed image data
    const std::vector<uint8_t> imageData(
        message.data.begin() + sizeof(compressionConfig), message.data.end());

    // Depth map decoding
    float depthQuantA, depthQuantB;

    // Read quantization parameters
    depthQuantA = compressionConfig.depthParam[0];
    depthQuantB = compressionConfig.depthParam[1];

    if (sensor_msgs::image_encodings::bitDepth(image_encoding) == 32) {
      cv::Mat decompressed;
      if (compression_format == "png") {
        try {
          // Decode image data
          decompressed = cv::imdecode(imageData, cv::IMREAD_UNCHANGED);
        } catch (cv::Exception& e) {
          ROS_ERROR("[compressed_image_receiver] %s", e.what());
          return sensor_msgs::Image::Ptr();
        }
      } else if (compression_format == "rvl") {
        const unsigned char* buffer = imageData.data();

        uint32_t cols, rows;
        memcpy(&cols, &buffer[0], 4);
        memcpy(&rows, &buffer[4], 4);
        if (rows == 0 || cols == 0) {
          ROS_ERROR_THROTTLE(
              1.0,
              "[compressed_image_receiver] Received malformed RVL-encoded image. Size %ix%i contains zero.",
              cols,
              rows);
          return sensor_msgs::Image::Ptr();
        }

        // Sanity check - the best compression ratio is 4x; we leave some buffer, so we
        // check whether the output image would not be more than 10x larger than the
        // compressed one. If it is, we probably received corrupted data. The condition
        // should be "numPixels * 2 > compressed.size() * 10" (because each pixel is 2
        // bytes), but to prevent overflow, we have canceled out the *2 from both sides
        // of the inequality.
        const auto numPixels = static_cast<uint64_t>(rows) * cols;
        if (numPixels > std::numeric_limits<int>::max() ||
            numPixels > static_cast<uint64_t>(imageData.size()) * 5) {
          ROS_ERROR_THROTTLE(
              1.0,
              "[compressed_image_receiver] Received malformed RVL-encoded image. It reports size %ux%u.",
              cols,
              rows);
          return sensor_msgs::Image::Ptr();
        }

        decompressed = cv::Mat(rows, cols, CV_16UC1);
        conversions::RvlCodec rvl;
        rvl.DecompressRVL(&buffer[8], decompressed.ptr<unsigned short>(), cols * rows);
      } else {
        return sensor_msgs::Image::Ptr();
      }

      size_t rows = decompressed.rows;
      size_t cols = decompressed.cols;

      if ((rows > 0) && (cols > 0)) {
        cv_ptr->image = cv::Mat(rows, cols, CV_32FC1);

        // Depth conversion
        cv::MatIterator_<float> itDepthImg = cv_ptr->image.begin<float>(),
                                itDepthImg_end = cv_ptr->image.end<float>();
        cv::MatConstIterator_<unsigned short> itInvDepthImg =
                                                  decompressed.begin<unsigned short>(),
                                              itInvDepthImg_end =
                                                  decompressed.end<unsigned short>();

        for (; (itDepthImg != itDepthImg_end) && (itInvDepthImg != itInvDepthImg_end);
             ++itDepthImg, ++itInvDepthImg) {
          // check for NaN & max depth
          if (*itInvDepthImg) {
            *itDepthImg = depthQuantA / ((float)*itInvDepthImg - depthQuantB);
          } else {
            *itDepthImg = std::numeric_limits<float>::quiet_NaN();
          }
        }

        // Publish message to user callback
        return cv_ptr->toImageMsg();
      }
    } else {
      // Decode raw image
      if (compression_format == "png") {
        try {
          cv_ptr->image = cv::imdecode(imageData, -1);
        } catch (cv::Exception& e) {
          ROS_ERROR("[compressed_image_receiver] %s", e.what());
          return sensor_msgs::Image::Ptr();
        }
      } else if (compression_format == "rvl") {
        const unsigned char* buffer = imageData.data();
        uint32_t cols, rows;
        memcpy(&cols, &buffer[0], 4);
        memcpy(&rows, &buffer[4], 4);
        cv_ptr->image = cv::Mat(rows, cols, CV_16UC1);
        conversions::RvlCodec rvl;
        rvl.DecompressRVL(&buffer[8], cv_ptr->image.ptr<unsigned short>(), cols * rows);
      } else {
        return sensor_msgs::Image::Ptr();
      }

      size_t rows = cv_ptr->image.rows;
      size_t cols = cv_ptr->image.cols;

      if ((rows > 0) && (cols > 0)) {
        // Publish message to user callback
        return cv_ptr->toImageMsg();
      }
    }
  }
  return sensor_msgs::Image::Ptr();
}

void CompressedImageReceiver::callback(
    const sensor_msgs::CompressedImageConstPtr& color,
    const sensor_msgs::CompressedImageConstPtr& depth,
    const sensor_msgs::ImageConstPtr& panoptic_ids,
    const semantic_inference_msgs::FeatureVectorStamped::ConstPtr& features,
    const semantic_inference_msgs::FeatureImage::ConstPtr& labels,
    const semantic_inference_msgs::FeatureVectorsStamped::ConstPtr& relations) {
  if (!maps_.init) {
    LOG(ERROR) << "Camera maps are not initialized. Cannot process images.";
    return;
  }
  if (!processing_enabled_) {
    ROS_WARN_THROTTLE(5.0, "Image processing paused via service toggle.");
    return;
  }

  if (!color || !depth) {
    LOG(ERROR) << "Received null color or depth image.";
    return;
  }
  auto decompressed_color =
      cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::BGR8);
  const auto decompressed_depth = decodeCompressedDepthImage(*depth);

  if (decompressed_color->image.cols != decompressed_depth->width ||
      decompressed_color->image.rows != decompressed_depth->height) {
    LOG(ERROR) << "color dimensions do not match depth dimensions: "
               << "Color: " << decompressed_color->image.cols << "x"
               << decompressed_color->image.rows
               << ", Depth: " << decompressed_depth->width << "x"
               << decompressed_depth->height;
    return;
  }
  const auto labels_image = boost::make_shared<sensor_msgs::Image>(labels->image);
  if (labels_image && (labels_image->width != decompressed_depth->width ||
                       labels_image->height != decompressed_depth->height)) {
    LOG(ERROR) << "label dimensions do not match depth dimensions";
    return;
  }

  if (!checkInputTimestamp(depth->header.stamp.toNSec())) {
    return;
  }

  auto packet =
      std::make_shared<ImageInputPacket>(color->header.stamp.toNSec(), sensor_id_);
  try {
    auto cv_depth = cv_bridge::toCvShare(decompressed_depth);
    packet->depth = cv_depth->image.clone();
    cv::remap(packet->depth, packet->depth, maps_.map1, maps_.map2, cv::INTER_LINEAR);

    // Convert depth to meters if it is in millimeters.
    if (cv_depth->image.type() == CV_16UC1) {
      packet->depth.convertTo(packet->depth, CV_32FC1, 0.001);
    } else if (cv_depth->image.type() != CV_32FC1) {
      LOG(ERROR) << "Unsupported depth image type: " << cv_depth->image.type();
      return;
    }
    packet->color = decompressed_color->image;
    cv::remap(packet->color, packet->color, maps_.map1, maps_.map2, cv::INTER_LINEAR);

    if (decompressed_color->encoding == sensor_msgs::image_encodings::BGR8) {
      cv::cvtColor(packet->color, packet->color, cv::COLOR_BGR2RGB);
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
