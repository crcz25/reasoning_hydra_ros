// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/input/enhanced_pointcloud_adaptor.h"

namespace hydra {

EnhancedPointcloudAdaptor::EnhancedPointcloudAdaptor(
    const sensor_msgs::PointCloud2& cloud) {
  for (const auto& field : cloud.fields) {
    if (field.name == "x") {
      VLOG(10) << "found x field: " << field;
      x_parser_ = initFloatParser(field);
    } else if (field.name == "y") {
      VLOG(10) << "found y field: " << field;
      y_parser_ = initFloatParser(field);
    } else if (field.name == "z") {
      VLOG(10) << "found z field: " << field;
      z_parser_ = initFloatParser(field);
    } else {
      VLOG(10) << "unknown field: " << field;
    }
  }
}

bool EnhancedPointcloudAdaptor::valid() const {
  return x_parser_ && y_parser_ && z_parser_;
}

cv::Vec3f EnhancedPointcloudAdaptor::position(const uint8_t* point_ptr) const {
  return cv::Vec3f(x_parser_(point_ptr), y_parser_(point_ptr), z_parser_(point_ptr));
}

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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& debug_pointcloud) {
  EnhancedPointcloudAdaptor adaptor(*cloud);
  if (!adaptor.valid()) {
    return false;
  }
  cv::Mat panoptic_ids_converted = cv::Mat::zeros(panoptic_ids.size(), CV_16UC1);
  panoptic_ids.convertTo(panoptic_ids_converted, CV_16UC1);
  cv::Size size(cloud->width, cloud->height);
  packet->points = cv::Mat(size, CV_32FC3);
  packet->colors = cv::Mat::zeros(size, CV_8UC3);
  packet->labels = cv::Mat::zeros(size, CV_8UC3);
  packet->valid =
      std::vector<std::vector<bool>>(size.height, std::vector<bool>(size.width, false));
  packet->features_mask = cv::Mat::zeros(size, CV_16UC1);
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

  for (uint32_t row = 0; row < cloud->height; ++row) {
    for (uint32_t col = 0; col < cloud->width; ++col) {
      const auto offset = row * cloud->row_step + col * cloud->point_step;
      const auto point_ptr = &cloud->data[offset];
      const auto point = adaptor.position(point_ptr);
      const auto point_cam =
          projectPointcloudToImage(camera_info, lidar2cam, point, undistort);
      // Check if the point is valid (z > 0, within image bounds)
      if (point_cam(2) > 0 && point_cam(0) >= 0 && point_cam(0) < camera_info->width &&
          point_cam(1) >= 0 && point_cam(1) < camera_info->height) {
        packet->valid[row][col] = true;
        // Assign color and label values
        packet->colors.at<cv::Vec3b>(row, col) =
            bilinearSampleVec3b(color, point_cam(0), point_cam(1));
        packet->labels.at<cv::Vec3b>(row, col) =
            bilinearSampleVec3b(label_image, point_cam(0), point_cam(1));
        packet->features_mask.value().at<uint16_t>(row, col) =
            bilinearSampleUint16(panoptic_ids_converted, point_cam(0), point_cam(1));

        // Add debug point to the point cloud
        if (debug_pointcloud) {
          pcl::PointXYZRGB debug_point;
          debug_point.x = point(0);
          debug_point.y = point(1);
          debug_point.z = point(2);
          debug_point.r = packet->colors.at<cv::Vec3b>(row, col)[2];
          debug_point.g = packet->colors.at<cv::Vec3b>(row, col)[1];
          debug_point.b = packet->colors.at<cv::Vec3b>(row, col)[0];
          debug_pointcloud->points.push_back(debug_point);
        }
      }
      packet->points.at<cv::Vec3f>(row, col) = point;
    }
  }
  return true;
}

Eigen::Vector3f projectPointcloudToImage(
    const sensor_msgs::CameraInfoConstPtr& camera_info,
    const PoseStatus& transform,
    const cv::Vec3f& point,
    const bool& undistort) {
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

  // Transform the point from world frame to camera frame
  Eigen::Vector4f pt_world(point(0), point(1), point(2), 1.0);
  Eigen::Vector4f pt_camera = transform.toMatrix().cast<float>() * pt_world;

  // Extract camera frame coordinates
  float x_cam = pt_camera(0);
  float y_cam = pt_camera(1);
  float z_cam = pt_camera(2);

  // Project the point to the image plane (camera intrinsics)
  float u = fx * (x_cam / z_cam) + cx;
  float v = fy * (y_cam / z_cam) + cy;

  if (undistort) {
    // Compute the radial distance from the principal point
    float r2 =
        (x_cam * x_cam + y_cam * y_cam) / (z_cam * z_cam);  // radial distance squared

    // Apply the equidistant distortion model
    float distortion_factor =
        (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2 + k4 * r2 * r2 * r2 * r2);

    // Apply the distortion to the u, v coordinates
    u = u * distortion_factor;
    v = v * distortion_factor;
  }

  return Eigen::Vector3f(u, v, z_cam);
}

cv::Vec3b bilinearSampleVec3b(const cv::Mat& img, float u, float v) {
  int x = static_cast<int>(std::floor(u));
  int y = static_cast<int>(std::floor(v));
  float dx = u - x;
  float dy = v - y;

  if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1) {
    return cv::Vec3b(0, 0, 0);
  }

  cv::Vec3f I00, I10, I01, I11;

  if (img.type() == CV_8UC3) {
    I00 = img.at<cv::Vec3b>(y, x);
    if (x + 1 >= img.cols) {
      I10 = img.at<cv::Vec3b>(y, x);
    } else {
      I10 = img.at<cv::Vec3b>(y, x + 1);
    }
    if (y + 1 >= img.rows) {
      I01 = img.at<cv::Vec3b>(y, x);
    } else {
      I01 = img.at<cv::Vec3b>(y + 1, x);
    }
    if (x + 1 >= img.cols || y + 1 >= img.rows) {
      I11 = img.at<cv::Vec3b>(y, x);
    } else {
      I11 = img.at<cv::Vec3b>(y + 1, x + 1);
    }
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported type for Vec3b sampling");
  }

  cv::Vec3f I0 = I00 * (1.0f - dx) + I10 * dx;
  cv::Vec3f I1 = I01 * (1.0f - dx) + I11 * dx;
  cv::Vec3f I = I0 * (1.0f - dy) + I1 * dy;

  return cv::Vec3b(cv::saturate_cast<uchar>(I[0]),
                   cv::saturate_cast<uchar>(I[1]),
                   cv::saturate_cast<uchar>(I[2]));
}

uint16_t bilinearSampleUint16(const cv::Mat& img, float u, float v) {
  int x = static_cast<int>(std::floor(u));
  int y = static_cast<int>(std::floor(v));
  float dx = u - x;
  float dy = v - y;

  if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1) {
    return 0;
  }

  float I00, I10, I01, I11;

  if (img.type() == CV_16UC1) {
    I00 = img.at<uint16_t>(y, x);
    if (x + 1 >= img.cols) {
      I10 = img.at<uint16_t>(y, x);
    } else {
      I10 = img.at<uint16_t>(y, x + 1);
    }
    if (y + 1 >= img.rows) {
      I01 = img.at<uint16_t>(y, x);
    } else {
      I01 = img.at<uint16_t>(y + 1, x);
    }
    if (x + 1 >= img.cols || y + 1 >= img.rows) {
      I11 = img.at<uint16_t>(y, x);
    } else {
      I11 = img.at<uint16_t>(y + 1, x + 1);
    }
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported type for uint16_t sampling");
  }

  float I0 = I00 * (1.0f - dx) + I10 * dx;
  float I1 = I01 * (1.0f - dx) + I11 * dx;
  float I = I0 * (1.0f - dy) + I1 * dy;

  return static_cast<uint16_t>(std::round(I));
}

int32_t bilinearSampleInt32(const cv::Mat& img, float u, float v) {
  int x = static_cast<int>(std::floor(u));
  int y = static_cast<int>(std::floor(v));
  float dx = u - x;
  float dy = v - y;

  if (x < 0 || x >= img.cols - 1 || y < 0 || y >= img.rows - 1) {
    return 0;
  }

  float I00, I10, I01, I11;

  if (img.type() == CV_32SC1) {
    I00 = img.at<int32_t>(y, x);
    I10 = img.at<int32_t>(y, x + 1);
    I01 = img.at<int32_t>(y + 1, x);
    I11 = img.at<int32_t>(y + 1, x + 1);
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported type for int32_t sampling");
  }

  float I0 = I00 * (1.0f - dx) + I10 * dx;
  float I1 = I01 * (1.0f - dx) + I11 * dx;
  float I = I0 * (1.0f - dy) + I1 * dy;

  return static_cast<int32_t>(std::round(I));
}

}  // namespace hydra
