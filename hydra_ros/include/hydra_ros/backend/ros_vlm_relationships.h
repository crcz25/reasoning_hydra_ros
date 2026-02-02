// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <geometry_msgs/Pose.h>
#include <hydra/backend/backend_module.h>
#include <hydra/common/shared_dsg_info.h>
#include <ros/ros.h>
#include <semantic_inference_msgs/FeatureVector.h>
#include <semantic_inference_msgs/FeatureVectorsStamped.h>
#include <vision_msgs/BoundingBox3D.h>
#include <vision_msgs/BoundingBox3DArray.h>

#include <memory>
#include <string>

#include "hydra_msgs/LabeledRelationships.h"
#include "hydra_msgs/VLMRelationship.h"
#include "hydra_msgs/VisualRelationshipsEncodings.h"

namespace hydra {

class RosVLMRelationships : public BackendModule::Sink {
 public:
  using Ptr = std::shared_ptr<RosVLMRelationships>;

  struct Config {
    std::string default_prompt = "What is the relationship between";
    int queue_size = 10;
  } const config;

  struct Request {
    std::string prompt;
    int room_id;
  };

  RosVLMRelationships(
      const ros::NodeHandle& nh,
      const Config& config,
      const InputQueue<BackendVLMLabelsInput::Ptr>::Ptr& vlm_labels_queue);
  virtual ~RosVLMRelationships() = default;

  void call(uint64_t timestamp_ns,
            const DynamicSceneGraph& graph,
            const kimera_pgmo::DeformationGraph& dgraph) override;

 private:
  bool handleRequest(hydra_msgs::VLMRelationship::Request& req,
                     hydra_msgs::VLMRelationship::Response& res);

  void labeledRelationshipsCallback(
      const hydra_msgs::LabeledRelationships::ConstPtr& relationships);

  void resetLabels(const DynamicSceneGraph& graph, uint64_t timestamp_ns);

  ros::NodeHandle nh_;
  ros::ServiceServer service_;
  ros::Subscriber labeled_relationships_sub_;
  ros::Publisher relationships_pub_;

  InputQueue<BackendVLMLabelsInput::Ptr>::Ptr vlm_labels_queue_;
  InputQueue<Request> request_queue_;
};
int extractNumber(const std::string& input);

void declare_config(RosVLMRelationships::Config& config);

}  // namespace hydra
