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

#include <config_utilities/config.h>
#include <config_utilities/parsing/ros.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <hydra/backend/backend_module.h>
#include <hydra/backend/update_frontiers_functor.h>
#include <hydra/backend/update_surface_places_functor.h>
#include <hydra/common/dsg_types.h>
#include <hydra/common/global_info.h>
#include <hydra/common/hydra_pipeline.h>
#include <hydra/frontend/frontend_module.h>
#include <hydra/loop_closure/loop_closure_module.h>
#include <hydra/navigation/navigation_module.h>
#include <hydra/navigation/object_search_module.h>
#include <hydra/reconstruction/reconstruction_module.h>
#include <pose_graph_tools_ros/conversions.h>
#include <ros/ros.h>

#include <memory>
#include <string>

#include "hydra_ros/backend/ros_backend_publisher.h"
#include "hydra_ros/backend/ros_vlm_relationships.h"
#include "hydra_ros/frontend/ros_frontend_publisher.h"
#include "hydra_ros/hydra_ros_pipeline.h"
#include "hydra_ros/navigation/ros_navigation_interface.h"

namespace hydra {

class HydraVisPipeline : public HydraPipeline {
 public:
  HydraVisPipeline(const ros::NodeHandle& nh, int robot_id);

  virtual ~HydraVisPipeline();

  void init() override;
  void loadGraph(const std::string& filepath);
  void publishVisualization();

 protected:
  virtual void initFrontend();
  virtual void initBackend(const ros::NodeHandle& bnh);
  virtual void initReconstruction();
  virtual void initLCD();
  virtual void initNavigation(const ros::NodeHandle& bnh);

 protected:
  const HydraRosConfig config_;
  ros::NodeHandle nh_;
  RosVLMRelationships::Ptr vlm_relationships_;
  RosNavigationInterface::Ptr navigation_interface_;
};

}  // namespace hydra
