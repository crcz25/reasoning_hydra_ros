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
#include <config_utilities/config_utilities.h>
#include <config_utilities/formatting/asl.h>
#include <config_utilities/logging/log_to_glog.h>
#include <config_utilities/parsing/ros.h>
#include <hydra/common/global_info.h>
#include <hydra_msgs/Load.h>
#include <std_srvs/Trigger.h>

#include <memory>

#include "hydra_ros/hydra_vis_pipeline.h"
#include "hydra_ros/utils/node_utilities.h"

class VisNodeWrapper {
 public:
  explicit VisNodeWrapper(ros::NodeHandle& nh) : nh_(nh) {}
  ~VisNodeWrapper() = default;

  void init(const int& robot_id, const std::string& filepath) {
    hydra_pipeline_ = std::make_unique<hydra::HydraVisPipeline>(nh_, robot_id);
    hydra_pipeline_->init();
    hydra_pipeline_->loadGraph(filepath);
  }
  void start() {
    hydra_pipeline_->start();
    _running = true;
    // Start the visualization publishing with a ros rate
  }
  void stop() {
    hydra_pipeline_->stop();
    _running = false;
  }

  void spin() { hydra::spinAndWait(nh_); }

  void spinPublisher() {
    ros::Rate rate(5);
    while (ros::ok() && _running) {
      hydra_pipeline_->publishVisualization();
      rate.sleep();
    }
  }

 private:
  std::unique_ptr<hydra::HydraVisPipeline> hydra_pipeline_;
  ros::ServiceServer save_graph_service_;
  ros::ServiceServer load_graph_service_;
  ros::NodeHandle nh_;
  std::atomic<bool> _running = false;
};

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "hydra_node");

  FLAGS_minloglevel = 3;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  config::Settings().setLogger("glog");
  config::Settings().print_width = 100;
  config::Settings().print_indent = 45;

  ros::NodeHandle nh("~");
  const int robot_id = nh.param<int>("robot_id", 0);
  std::string filepath = nh.param<std::string>("graph_filepath", "");
  VisNodeWrapper node_wrapper(nh);
  node_wrapper.init(robot_id, filepath);
  node_wrapper.start();
  // Start the visualization publishing in a separate thread
  std::thread vis_thread(&VisNodeWrapper::spinPublisher, &node_wrapper);
  vis_thread.detach();  // Detach the thread to run independently
  node_wrapper.spin();
  node_wrapper.stop();
  hydra::GlobalInfo::exit();

  return 0;
}
