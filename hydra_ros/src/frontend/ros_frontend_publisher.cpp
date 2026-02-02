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
#include "hydra_ros/frontend/ros_frontend_publisher.h"

#include <hydra/common/global_info.h>
#include <kimera_pgmo_msgs/KimeraPgmoMeshDelta.h>
#include <kimera_pgmo_ros/conversion/mesh_delta_conversion.h>
#include <pose_graph_tools_msgs/PoseGraph.h>
#include <pose_graph_tools_ros/conversions.h>
#include <sensor_msgs/PointCloud2.h>

namespace hydra {

using kimera_pgmo_msgs::KimeraPgmoMeshDelta;
using pose_graph_tools_msgs::PoseGraph;

void declare_config(RosFrontendPublisher::Config& config) {
  using namespace config;
  name("RosFrontendPublisher::Config");
  field(config.label_colormap, "label_colormap");
  field(config.color_by_label, "color_by_label");
}

RosFrontendPublisher::RosFrontendPublisher(const ros::NodeHandle& node_handle,
                                           const Config& config)
    : nh_(node_handle), config(config::checkValid(config)) {
  const auto odom_frame = GlobalInfo::instance().getFrames().odom;
  dsg_sender_.reset(new DsgSender(nh_, odom_frame, "frontend", false));
  mesh_graph_pub_ = nh_.advertise<PoseGraph>("mesh_graph_incremental", 100, true);
  mesh_update_pub_ = nh_.advertise<KimeraPgmoMeshDelta>("full_mesh_update", 100, true);
  semantic_pcl_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("semantic_pcl", 100, true);
  if (!config.label_colormap.empty()) {
    colormap_ = SemanticColorMap::fromCsv(config.label_colormap);
    if (!colormap_) {
      ROS_WARN_STREAM("Unable to load colormap from " << config.label_colormap);
    } else {
      color_by_label_ = config.color_by_label;
      toggle_service_ = nh_.advertiseService(
          "color_by_label", &RosFrontendPublisher::handleService, this);
    }
  }
}

void RosFrontendPublisher::call(uint64_t timestamp_ns,
                                const DynamicSceneGraph& graph,
                                const BackendInput& backend_input) {
  if (backend_input.deformation_graph) {
    auto msg = pose_graph_tools::toMsg(*backend_input.deformation_graph);
    msg.header.stamp.fromNSec(timestamp_ns);
    mesh_graph_pub_.publish(msg);
  }

  if (backend_input.mesh_update) {
    mesh_update_pub_.publish(
        kimera_pgmo::conversions::toRosMsg(*backend_input.mesh_update, timestamp_ns));
  }

  const auto& pointcloud = backend_input.getPointCloud();
  if (pointcloud) {
    sensor_msgs::PointCloud2 msg;

    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr semantic_pcl(
        new pcl::PointCloud<pcl::PointXYZRGBL>());
    if (hydra::toPcl(pointcloud, semantic_pcl)) {
      if (color_by_label_ && colormap_) {
        for (auto& point : semantic_pcl->points) {
          const auto color = colormap_->getColorFromLabel(point.label);
          point.r = color.r;
          point.g = color.g;
          point.b = color.b;
        }
      }
      pcl::toROSMsg(*semantic_pcl, msg);
      msg.header.stamp.fromNSec(timestamp_ns);
      msg.header.frame_id = GlobalInfo::instance().getFrames().odom;
      semantic_pcl_pub_.publish(msg);
    }
  }

  ros::Time stamp;
  stamp.fromNSec(timestamp_ns);
  dsg_sender_->sendGraph(graph, stamp);
}

bool RosFrontendPublisher::handleService(std_srvs::SetBool::Request& req,
                                         std_srvs::SetBool::Response& res) {
  color_by_label_ = req.data;
  res.success = true;
  return true;
}

}  // namespace hydra
