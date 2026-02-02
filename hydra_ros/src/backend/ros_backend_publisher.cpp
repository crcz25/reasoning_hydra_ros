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
#include "hydra_ros/backend/ros_backend_publisher.h"

namespace hydra {

using hydra_msgs::ActiveObjectRelationships;
using kimera_pgmo::DeformationGraph;
using kimera_pgmo::KimeraPgmoConfig;
using kimera_pgmo_msgs::KimeraPgmoMesh;
using pose_graph_tools_msgs::PoseGraph;
using visualization_msgs::Marker;

RosBackendPublisher::RosBackendPublisher(const ros::NodeHandle& nh) : nh_(nh) {
  mesh_mesh_edges_pub_ =
      nh_.advertise<Marker>("deformation_graph_mesh_mesh", 10, false);
  pose_mesh_edges_pub_ =
      nh_.advertise<Marker>("deformation_graph_pose_mesh", 10, false);
  pose_graph_pub_ = nh_.advertise<PoseGraph>("pose_graph", 10, false);
  edges_pub_ =
      nh_.advertise<ActiveObjectRelationships>("active_object_edges", 10, false);

  double separation = 0.0;
  nh_.getParam("min_mesh_separation_s", separation);
  const auto map_frame = GlobalInfo::instance().getFrames().map;
  dsg_sender_.reset(new hydra::DsgSender(nh_, map_frame, "backend", false, separation));
}

void RosBackendPublisher::call(uint64_t timestamp_ns,
                               const DynamicSceneGraph& graph,
                               const DeformationGraph& dgraph) {
  ros::Time stamp;
  stamp.fromNSec(timestamp_ns);
  dsg_sender_->sendGraph(graph, stamp);

  if (pose_graph_pub_.getNumSubscribers() > 0) {
    publishPoseGraph(graph, dgraph);
  }

  if (mesh_mesh_edges_pub_.getNumSubscribers() > 0 ||
      pose_mesh_edges_pub_.getNumSubscribers() > 0) {
    publishDeformationGraphViz(dgraph, timestamp_ns);
  }

  if (edges_pub_.getNumSubscribers() > 0) {
    publishActiveObjectEdges(graph, timestamp_ns);
  }
}

void RosBackendPublisher::publishPoseGraph(const DynamicSceneGraph& graph,
                                           const DeformationGraph& dgraph) const {
  const auto& prefix = GlobalInfo::instance().getRobotPrefix();
  const auto& agent = graph.getLayer(DsgLayers::AGENTS, prefix.key);

  std::map<size_t, std::vector<size_t>> id_timestamps;
  id_timestamps[prefix.id] = std::vector<size_t>();
  auto& times = id_timestamps[prefix.id];
  for (const auto& node : agent.nodes()) {
    times.push_back(node->timestamp.value().count());
  }

  const auto& pose_graph = *dgraph.getPoseGraph(id_timestamps);
  pose_graph_pub_.publish(pose_graph_tools::toMsg(pose_graph));
}

void RosBackendPublisher::publishDeformationGraphViz(const DeformationGraph& dgraph,
                                                     size_t timestamp_ns) const {
  ros::Time stamp;
  stamp.fromNSec(timestamp_ns);

  Marker mm_edges_msg;
  Marker pm_edges_msg;
  kimera_pgmo::fillDeformationGraphMarkers(dgraph,
                                           stamp,
                                           mm_edges_msg,
                                           pm_edges_msg,
                                           GlobalInfo::instance().getFrames().map);

  if (!mm_edges_msg.points.empty()) {
    mesh_mesh_edges_pub_.publish(mm_edges_msg);
  }
  if (!pm_edges_msg.points.empty()) {
    pose_mesh_edges_pub_.publish(pm_edges_msg);
  }
}

void RosBackendPublisher::publishActiveObjectEdges(const DynamicSceneGraph& graph,
                                                   size_t timestamp_ns) const {
  ros::Time stamp;
  stamp.fromNSec(timestamp_ns);

  ActiveObjectRelationships edges_msg;
  edges_msg.header.stamp = stamp;
  edges_msg.header.frame_id = GlobalInfo::instance().getFrames().odom;

  const auto& objects_layer = graph.getLayer(DsgLayers::OBJECTS);

  std::vector<NodeId> ordered_ids;
  std::unordered_map<NodeId, uint32_t> id_to_index;

  auto addObjectIfNeeded = [&](const NodeId& id, const ObjectNodeAttributes& attrs) {
    if (id_to_index.find(id) != id_to_index.end()) {
      return;
    }

    vision_msgs::BoundingBox3D box;
    box.center.position.x = attrs.bounding_box.world_P_center.x();
    box.center.position.y = attrs.bounding_box.world_P_center.y();
    box.center.position.z = attrs.bounding_box.world_P_center.z();

    Eigen::Quaternionf q(attrs.bounding_box.world_R_center.matrix());
    box.center.orientation.x = q.x();
    box.center.orientation.y = q.y();
    box.center.orientation.z = q.z();
    box.center.orientation.w = q.w();

    box.size.x = attrs.bounding_box.dimensions.x();
    box.size.y = attrs.bounding_box.dimensions.y();
    box.size.z = attrs.bounding_box.dimensions.z();

    uint32_t index = static_cast<uint32_t>(ordered_ids.size());
    ordered_ids.push_back(id);
    id_to_index[id] = index;
    edges_msg.object_boxes.push_back(box);
  };

  for (const auto& [edge_key, edge] : objects_layer.edges()) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) {
      continue;
    }
    const auto& source_node = graph.getNode(edge.source);
    const auto& source_bb = source_node.attributes<ObjectNodeAttributes>().bounding_box;
    const auto& target_node = graph.getNode(edge.target);
    const auto& target_bb = target_node.attributes<ObjectNodeAttributes>().bounding_box;
    if (!source_bb.isValid() || !target_bb.isValid()) {
      continue;
    }

    addObjectIfNeeded(edge.source, source_node.attributes<ObjectNodeAttributes>());
    addObjectIfNeeded(edge.target, target_node.attributes<ObjectNodeAttributes>());

    // Add edge indices
    edges_msg.object_ids.push_back(id_to_index[edge.source]);
    edges_msg.object_ids.push_back(id_to_index[edge.target]);
  }

  if (!edges_msg.object_boxes.empty()) {
    edges_pub_.publish(edges_msg);
  }
}

}  // namespace hydra
