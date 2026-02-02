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

#include "hydra_ros/visualizer/hydra_visualizer.h"

#include <config_utilities/config.h>
#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>
#include <hydra/utils/timing_utilities.h>

namespace hydra {

void declare_config(HydraVisualizerConfig& config) {
  using namespace config;
  name("HydraVisualizerConfig");
  field(config.load_graph, "load_graph");
  field(config.use_zmq, "use_zmq");
  field(config.scene_graph_filepath, "scene_graph_filepath");
  field(config.visualizer_ns, "visualizer_ns");
  field(config.output_path, "output_path");
  field(config.zmq_url, "zmq_url");
  field(config.zmq_num_threads, "zmq_num_threads");
  field(config.plugins, "plugins");
}

HydraVisualizer::HydraVisualizer(const ros::NodeHandle& nh) : nh_(nh) {
  const auto yaml_node = config::internal::rosToYaml(nh);
  VLOG(5) << "ROS Config: " << std::endl << yaml_node;

  config_ = config::fromRos<HydraVisualizerConfig>(nh);
  // ROS_INFO_STREAM("Config: " << std::endl << config_);

  visualizer_.reset(new DsgVisualizer(nh_));
  for (auto&& [name, config] : config_.plugins) {
    auto plugin = config.create(nh_, name);
    if (plugin) {
      visualizer_->addPlugin(name, std::move(plugin));
    }
  }

  if (!config_.output_path.empty()) {
    size_log_file_.reset(
        new std::ofstream(config_.output_path + "/dsg_message_sizes.csv"));
    *size_log_file_ << "time_ns,bytes" << std::endl;
  }

  nav_path_sub_ =
      nh_.subscribe("navigation_path", 1, &HydraVisualizer::navPathCallback, this);

  evaluating_objects_sub_ = nh_.subscribe(
      "evaluating_objects", 1, &HydraVisualizer::evaluatingObjectsCallback, this);
}

HydraVisualizer::~HydraVisualizer() {
  std::cout << "timing stats: "
            << hydra::timing::ElapsedTimeRecorder::instance().getStats("receive_dsg")
            << std::endl;
  std::cout << "mesh timing stats: "
            << hydra::timing::ElapsedTimeRecorder::instance().getStats("receive_mesh")
            << std::endl;
}

void HydraVisualizer::loadGraph() {
  ROS_INFO_STREAM("Loading dsg from: " << config_.scene_graph_filepath);
  auto dsg = hydra::DynamicSceneGraph::load(config_.scene_graph_filepath);
  ROS_INFO_STREAM("Loaded dsg: " << dsg->numNodes() << " nodes, " << dsg->numEdges()
                                 << " edges, has mesh? "
                                 << (dsg->hasMesh() ? "yes" : "no"));
  visualizer_->setGraph(dsg);
}

bool HydraVisualizer::handleReload(std_srvs::Empty::Request&,
                                   std_srvs::Empty::Response&) {
  loadGraph();
  return true;
}

bool HydraVisualizer::handleRedraw(std_srvs::Empty::Request&,
                                   std_srvs::Empty::Response&) {
  // TODO(nathan) this is technically safe, but not super ideal
  visualizer_->setGraphUpdated();
  visualizer_->redraw();
  return true;
}

bool HydraVisualizer::resetObjectSearch(std_srvs::Empty::Request&,
                                        std_srvs::Empty::Response&) {
  visualizer_->setObjectSearch(false);
  visualizer_->setObjectsEvaluation(false);
  visualizer_->deleteNavigationMarkers();
  visualizer_->redraw();
  return true;
}

bool HydraVisualizer::publishFinalMesh(std_srvs::Empty::Request&,
                                       std_srvs::Empty::Response&) {
  if (!visualizer_->graphIsSet()) {
    ROS_WARN("No graph set, cannot publish final mesh!");
    return false;
  }
  return visualizer_->publishFinalMesh();
}

void HydraVisualizer::navPathCallback(const hydra_msgs::Navigation::ConstPtr& msg) {
  if (!visualizer_->graphIsSet()) {
    return;
  }
  NavigationOutput nav;
  std::set<NodeId> object_search_nodes;
  for (const auto& nav_path : msg->navigation_paths) {
    NavigationPath path;
    path.object_id = nav_path.object_id;
    path.target_id = nav_path.target_id;
    object_search_nodes.insert(nav_path.object_id);
    object_search_nodes.insert(nav_path.target_id);
    path.object_label = nav_path.object_label;
    path.target_label = nav_path.target_label;
    path.explanation = nav_path.explanation;
    for (const auto& point : nav_path.agent_to_target) {
      path.agent_to_target.push_back(Eigen::Vector3d(point.x, point.y, point.z));
    }
    for (const auto& point : nav_path.target_to_object) {
      path.target_to_object.push_back(Eigen::Vector3d(point.x, point.y, point.z));
    }
    nav.paths.push_back(path);
  }

  if (nav.paths.empty()) {
    return;
  }
  visualizer_->setObjectsEvaluation(false);
  visualizer_->setObjectSearch(true, object_search_nodes);
  visualizer_->drawNavigation(nav);
  visualizer_->redraw();
}

void HydraVisualizer::evaluatingObjectsCallback(
    const std_msgs::UInt64MultiArray::ConstPtr& msg) {
  if (!visualizer_->graphIsSet()) {
    return;
  }
  std::set<NodeId> evaluating_objects;
  for (const auto& id : msg->data) {
    evaluating_objects.insert(id);
  }
  visualizer_->setObjectSearch(false);
  visualizer_->setObjectsEvaluation(true, evaluating_objects);
  visualizer_->redraw();
}

void HydraVisualizer::spinRos() {
  receiver_.reset(new DsgReceiver(nh_, [&](const ros::Time& stamp, size_t bytes) {
    if (size_log_file_) {
      *size_log_file_ << stamp.toNSec() << "," << bytes << std::endl;
    }
  }));

  bool graph_set = false;

  ros::WallRate r(10);
  while (ros::ok()) {
    ros::spinOnce();

    if (receiver_ && receiver_->updated()) {
      if (!receiver_->graph()) {
        r.sleep();
        continue;
      }
      if (!graph_set) {
        visualizer_->setGraph(receiver_->graph());
        graph_set = true;
      } else {
        visualizer_->setGraphUpdated();
      }
      visualizer_->setPromptEmbedding(receiver_->promptEmbedding());
      visualizer_->redraw();
      receiver_->clearUpdated();
    }

    r.sleep();
  }
}

void HydraVisualizer::spinFile() {
  if (config_.scene_graph_filepath.empty()) {
    LOG(ERROR) << "Scene graph filepath invalid!";
    return;
  }

  loadGraph();

  reload_service_ =
      nh_.advertiseService("reload", &HydraVisualizer::handleReload, this);
  visualizer_->start();

  // Still publish updates if they need to be e.g. redrawn.
  ros::WallRate r(5);
  while (ros::ok()) {
    ros::spinOnce();
    visualizer_->setGraphUpdated();
    visualizer_->redraw();
    r.sleep();
  }
}

void HydraVisualizer::spinZmq() {
  zmq_receiver_.reset(
      new spark_dsg::ZmqReceiver(config_.zmq_url, config_.zmq_num_threads));

  bool graph_set = false;
  while (ros::ok()) {
    ros::spinOnce();

    // we always receive all messages
    if (!zmq_receiver_->recv(config_.zmq_poll_time_ms, true)) {
      continue;
    }

    auto graph = zmq_receiver_->graph();
    if (!graph) {
      continue;
    }

    if (!graph_set) {
      visualizer_->setGraph(graph);
      graph_set = true;
    } else {
      visualizer_->setGraphUpdated();
    }

    visualizer_->redraw();
  }
}

void HydraVisualizer::spin() {
  ROS_DEBUG("Visualizer running");
  redraw_service_ =
      nh_.advertiseService("redraw", &HydraVisualizer::handleRedraw, this);

  reset_object_search_coloring_service_ = nh_.advertiseService(
      "reset_object_search_coloring", &HydraVisualizer::resetObjectSearch, this);
  final_mesh_publish_service_ = nh_.advertiseService(
      "publish_final_mesh", &HydraVisualizer::publishFinalMesh, this);

  if (config_.load_graph) {
    spinFile();
    return;
  }

  if (config_.use_zmq) {
    spinZmq();
  } else {
    spinRos();
  }
}

void HydraVisualizer::addPlugin(DsgVisualizerPlugin::Ptr plugin) {
  visualizer_->addPlugin("", std::move(plugin));
}

void HydraVisualizer::clearPlugins() { visualizer_->clearPlugins(); }

}  // namespace hydra
