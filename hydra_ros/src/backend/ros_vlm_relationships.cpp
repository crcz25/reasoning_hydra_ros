// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/backend/ros_vlm_relationships.h"

namespace hydra {

int extractNumber(const std::string& input) {
  std::string numberStr;
  for (char c : input) {
    if (std::isdigit(c)) {
      numberStr += c;
    } else if (!numberStr.empty()) {
      break;  // Stop once digits end (assuming only one number)
    }
  }

  if (!numberStr.empty()) {
    return std::stoi(numberStr);
  }

  throw std::runtime_error("No number found in the string");
}

void declare_config(RosVLMRelationships::Config& conf) {
  using namespace config;
  name("RosVLMRelationshipsConfig");
  field(conf.default_prompt, "default_prompt");
  field(conf.queue_size, "queue_size");
}

RosVLMRelationships::RosVLMRelationships(
    const ros::NodeHandle& nh,
    const Config& config,
    const InputQueue<BackendVLMLabelsInput::Ptr>::Ptr& vlm_labels_queue)
    : config(config), nh_(nh), vlm_labels_queue_(vlm_labels_queue) {
  service_ = nh_.advertiseService(
      "vlm_relationships", &RosVLMRelationships::handleRequest, this);
  labeled_relationships_sub_ =
      nh_.subscribe("labeled_relationships",
                    config.queue_size,
                    &RosVLMRelationships::labeledRelationshipsCallback,
                    this);
  relationships_pub_ = nh_.advertise<hydra_msgs::VisualRelationshipsEncodings>(
      "visual_relationships_encodings", config.queue_size, false);
}

bool RosVLMRelationships::handleRequest(hydra_msgs::VLMRelationship::Request& req,
                                        hydra_msgs::VLMRelationship::Response& res) {
  // Push the request to the queue
  if (relationships_pub_.getNumSubscribers() <= 0) {
    res.success = false;
    return true;
  }
  res.success = true;
  if (!request_queue_.push(
          {!req.prompt.empty() ? req.prompt : config.default_prompt, req.room})) {
    LOG(WARNING) << "Request queue is full!";
    res.success = false;
  }
  return true;
}

void RosVLMRelationships::call(uint64_t timestamp_ns,
                               const DynamicSceneGraph& graph,
                               const kimera_pgmo::DeformationGraph& dgraph) {
  if (relationships_pub_.getNumSubscribers() <= 0 || !request_queue_.poll()) {
    return;
  }
  const auto& request = request_queue_.front();
  if (request.prompt == "reset" || request.prompt == "RESET") {
    resetLabels(graph, timestamp_ns);
    request_queue_.pop();
    return;
  }
  hydra_msgs::VisualRelationshipsEncodings relationship_encodings;
  const auto& object_layer = graph.getLayer(DsgLayers::OBJECTS);
  const auto& object_layer_edges = object_layer.edges();
  for (const auto& [edge_key, edge] : object_layer_edges) {
    const auto& source_node = object_layer.getNode(edge.source);
    const auto& target_node = object_layer.getNode(edge.target);
    if (request.room_id >= 0) {
      const auto& source_parents = source_node.parents();
      const auto& target_parents = target_node.parents();
      if (source_parents.empty() || target_parents.empty()) {
        continue;
      }
      if (!graph.hasNode(*source_parents.begin()) ||
          !graph.hasNode(*target_parents.begin())) {
        continue;
      }
      const auto& source_parent = graph.getNode(*source_parents.begin());
      const auto& target_parent = graph.getNode(*target_parents.begin());
      const auto& source_room_id = source_parent.getParent();
      const auto& target_room_id = target_parent.getParent();
      if (!source_room_id || !target_room_id) {
        continue;
      }
      int source_room_number = extractNumber(
          graph.getNode(*source_room_id).attributes<RoomNodeAttributes>().name);
      int target_room_number = extractNumber(
          graph.getNode(*target_room_id).attributes<RoomNodeAttributes>().name);
      if (source_room_number != request.room_id &&
          target_room_number != request.room_id) {
        continue;
      }
    }
    // Add the relationship to the message
    semantic_inference_msgs::FeatureVector source_feature_msg;
    semantic_inference_msgs::FeatureVector target_feature_msg;
    const Eigen::MatrixXf& source_feature = edge.info->feature(edge.source);
    const Eigen::MatrixXf& target_feature = edge.info->feature(edge.target);
    if (!source_feature.size() || !target_feature.size()) {
      continue;
    }
    source_feature_msg.rows = source_feature.rows();
    source_feature_msg.cols = source_feature.cols();
    target_feature_msg.rows = target_feature.rows();
    target_feature_msg.cols = target_feature.cols();

    // Map the matrix to the message (vector<float>)
    for (int i = 0; i < source_feature.rows(); ++i) {
      for (int j = 0; j < source_feature.cols(); ++j) {
        source_feature_msg.data.push_back(source_feature(i, j));
        target_feature_msg.data.push_back(target_feature(i, j));
      }
    }
    relationship_encodings.features.feature.push_back(source_feature_msg);
    relationship_encodings.features.feature.push_back(target_feature_msg);

    const auto& source_attributes = source_node.attributes<SemanticNodeAttributes>();
    const auto& target_attributes = target_node.attributes<SemanticNodeAttributes>();

    // Nodes classes
    std::string label_source = source_attributes.name;
    std::string label_target = target_attributes.name;
    label_source =
        label_source.empty() ? NodeSymbol(source_node.id).getLabel() : label_source;
    label_target =
        label_target.empty() ? NodeSymbol(target_node.id).getLabel() : label_target;

    // Nodes poses
    geometry_msgs::Pose source_point;
    geometry_msgs::Pose target_point;
    source_point.position.x = source_attributes.position.x();
    source_point.position.y = source_attributes.position.y();
    source_point.position.z = source_attributes.position.z();
    target_point.position.x = target_attributes.position.x();
    target_point.position.y = target_attributes.position.y();
    target_point.position.z = target_attributes.position.z();

    // Nodes BBs
    vision_msgs::BoundingBox3D source_bb;
    vision_msgs::BoundingBox3D target_bb;

    source_bb.center.position.x = source_attributes.bounding_box.world_P_center.x();
    source_bb.center.position.y = source_attributes.bounding_box.world_P_center.y();
    source_bb.center.position.z = source_attributes.bounding_box.world_P_center.z();
    if (source_attributes.bounding_box.hasRotation()) {
      Eigen::Quaternionf q(source_attributes.bounding_box.world_R_center);
      source_bb.center.orientation.x = q.x();
      source_bb.center.orientation.y = q.y();
      source_bb.center.orientation.z = q.z();
      source_bb.center.orientation.w = q.w();
    } else {
      source_bb.center.orientation.w = 1.0;
    }
    source_bb.size.x = source_attributes.bounding_box.dimensions.x();
    source_bb.size.y = source_attributes.bounding_box.dimensions.y();
    source_bb.size.z = source_attributes.bounding_box.dimensions.z();

    target_bb.center.position.x = target_attributes.bounding_box.world_P_center.x();
    target_bb.center.position.y = target_attributes.bounding_box.world_P_center.y();
    target_bb.center.position.z = target_attributes.bounding_box.world_P_center.z();
    if (target_attributes.bounding_box.hasRotation()) {
      Eigen::Quaternionf q(target_attributes.bounding_box.world_R_center);
      target_bb.center.orientation.x = q.x();
      target_bb.center.orientation.y = q.y();
      target_bb.center.orientation.z = q.z();
      target_bb.center.orientation.w = q.w();
    } else {
      target_bb.center.orientation.w = 1.0;
    }
    target_bb.size.x = target_attributes.bounding_box.dimensions.x();
    target_bb.size.y = target_attributes.bounding_box.dimensions.y();
    target_bb.size.z = target_attributes.bounding_box.dimensions.z();

    relationship_encodings.object_bounding_boxes.boxes.push_back(source_bb);
    relationship_encodings.object_bounding_boxes.boxes.push_back(target_bb);
    relationship_encodings.object_bounding_boxes.boxes.push_back(target_bb);
    relationship_encodings.object_bounding_boxes.boxes.push_back(source_bb);

    relationship_encodings.object_poses.push_back(source_point);
    relationship_encodings.object_poses.push_back(target_point);
    relationship_encodings.object_poses.push_back(target_point);
    relationship_encodings.object_poses.push_back(source_point);

    relationship_encodings.object_classes.push_back(label_source);
    relationship_encodings.object_classes.push_back(label_target);
    relationship_encodings.object_classes.push_back(label_target);
    relationship_encodings.object_classes.push_back(label_source);

    relationship_encodings.features.ids.push_back(edge.source);
    relationship_encodings.features.ids.push_back(edge.target);
    relationship_encodings.features.ids.push_back(edge.target);
    relationship_encodings.features.ids.push_back(edge.source);
  }

  relationship_encodings.prompt = request.prompt;
  request_queue_.pop();
  if (relationship_encodings.features.ids.empty()) {
    return;
  }
  relationship_encodings.features.header.stamp = ros::Time::now();
  relationships_pub_.publish(relationship_encodings);
}

void RosVLMRelationships::labeledRelationshipsCallback(
    const hydra_msgs::LabeledRelationships::ConstPtr& relationships) {
  // Loop over the labels of the relationships and add them to the queue
  auto vlm_labels = std::make_shared<VLMLabels>();
  for (size_t i = 0; i < relationships->labels.size(); ++i) {
    // Add the relationship to the queue
    vlm_labels->labels.push_back(relationships->labels[i]);
    vlm_labels->edge_ids.push_back(std::make_pair(relationships->node_ids[2 * i],
                                                  relationships->node_ids[2 * i + 1]));
  }
  // Push the relationships to the queue
  if (vlm_labels->labels.empty()) {
    return;
  }
  auto vlm_labels_input = std::make_shared<BackendVLMLabelsInput>();
  vlm_labels_input->vlm_labels = vlm_labels;
  vlm_labels_input->timestamp_ns = relationships->header.stamp.toNSec();
  if (!vlm_labels_queue_->push(vlm_labels_input)) {
    LOG(WARNING) << "VLM labels queue is full!";
  }
}

void RosVLMRelationships::resetLabels(const DynamicSceneGraph& graph,
                                      uint64_t timestamp_ns) {
  auto vlm_labels = std::make_shared<VLMLabels>();
  const auto& object_layer = graph.getLayer(DsgLayers::OBJECTS);
  const auto& object_layer_edges = object_layer.edges();
  for (const auto& [edge_key, edge] : object_layer_edges) {
    vlm_labels->labels.push_back("");
    vlm_labels->labels.push_back("");
    vlm_labels->edge_ids.push_back(std::make_pair(edge.source, edge.target));
    vlm_labels->edge_ids.push_back(std::make_pair(edge.target, edge.source));
  }
  auto vlm_labels_input = std::make_shared<BackendVLMLabelsInput>();
  vlm_labels_input->vlm_labels = vlm_labels;
  vlm_labels_input->timestamp_ns = timestamp_ns;
  if (!vlm_labels_queue_->push(vlm_labels_input)) {
    LOG(WARNING) << "VLM labels queue is full!";
  }
}
}  // namespace hydra
