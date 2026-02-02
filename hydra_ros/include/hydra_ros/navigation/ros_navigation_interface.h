// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <hydra/common/global_info.h>
#include <hydra/common/input_queue.h>
#include <hydra/common/shared_dsg_info.h>
#include <hydra/navigation/navigation_module.h>
#include <hydra/navigation/object_search_module.h>
#include <hydra_ros/NavigationConfig.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <semantic_inference_msgs/FeatureVector.h>
#include <semantic_inference_msgs/NavigationPromptsEmbeddings.h>
#include <std_msgs/ColorRGBA.h>
#include <std_srvs/Empty.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "hydra_msgs/GetNavigation.h"
#include "hydra_msgs/Navigation.h"
#include "hydra_msgs/NavigationPath.h"
#include "hydra_msgs/ObjectSearch.h"
#include "hydra_msgs/ObjectSearchPair.h"
#include "hydra_ros/utils/dsg_streaming_interface.h"

namespace hydra {

class RosNavigationInterface {
 public:
  using Ptr = std::unique_ptr<RosNavigationInterface>;

  struct Config {
    bool enable_one_by_one_navigation = false;
    bool always_publish_waypoints = false;
  } config;

  RosNavigationInterface(
      const Config& config,
      const ros::NodeHandle& nh,
      const ros::NodeHandle& backend_nh,
      const InputQueue<NavigationInput::Ptr>::Ptr& navigation_input_queue,
      const InputQueue<ObjectSearchInput::Ptr>::Ptr& object_search_input_queue,
      const InputQueue<NavigationOutput>::Ptr& navigation_output_queue,
      const InputQueue<ObjectSearchOutput::Ptr>::Ptr& object_search_output_queue,
      const NavigationModule::Ptr& navigation_module,
      const ObjectSearchModule::Ptr object_search_module);
  ~RosNavigationInterface() = default;

  void spinRos();
  void spin();

 private:
  bool handleNavigationRequest(hydra_msgs::GetNavigation::Request& req,
                               hydra_msgs::GetNavigation::Response& res);

  bool handleFindNext(std_srvs::Empty::Request& req, std_srvs::Empty::Response& res);

  bool handlePublishWaypoints(std_srvs::Empty::Request& req,
                              std_srvs::Empty::Response& res);

  bool handleClearNavigation(std_srvs::Empty::Request& req,
                             std_srvs::Empty::Response& res);

  void callbackObjectSearch(
      const semantic_inference_msgs::NavigationPromptsEmbeddings::ConstPtr& msg);

  void configCallback(hydra_ros::NavigationConfig& cfg, uint32_t level);

  void publishNavigation(const NavigationOutput& output);

  void publishObjectSearch(const ObjectSearchOutput::Ptr& output) const;

  struct NodeHandlers {
    ros::NodeHandle nh;
    ros::NodeHandle backend_nh;
    NodeHandlers(const ros::NodeHandle& nh, const ros::NodeHandle& backend_nh)
        : nh(nh), backend_nh(backend_nh) {}
  } node_handlers_;

  struct Services {
    ros::ServiceServer navigation;
    ros::ServiceServer find_next;
    ros::ServiceServer publish_waypoints;
    ros::ServiceServer clear_navigation;
  } services_;

  ros::Subscriber object_search_sub_;

  struct Publishers {
    ros::Publisher navigation_output;
    ros::Publisher navigation_goals;
    ros::Publisher object_search_output;
  } publishers_;

  std::shared_ptr<dynamic_reconfigure::Server<hydra_ros::NavigationConfig>>
      dyn_config_server_;
  dynamic_reconfigure::Server<hydra_ros::NavigationConfig>::CallbackType
      dyn_config_callback_;

  struct Threads {
    std::unique_ptr<std::thread> spin;
    std::unique_ptr<std::thread> spin_ros;
  } threads_;

  std::atomic<bool> should_shutdown_{false};

  struct InputQueues {
    InputQueue<NavigationInput::Ptr>::Ptr navigation;
    InputQueue<ObjectSearchInput::Ptr>::Ptr object_search;
    InputQueues(const InputQueue<NavigationInput::Ptr>::Ptr& navigation,
                const InputQueue<ObjectSearchInput::Ptr>::Ptr& object_search)
        : navigation(navigation), object_search(object_search) {}
  } input_queues_;

  struct OutputQueues {
    InputQueue<NavigationOutput>::Ptr navigation;
    InputQueue<ObjectSearchOutput::Ptr>::Ptr object_search;
    OutputQueues(const InputQueue<NavigationOutput>::Ptr& navigation,
                 const InputQueue<ObjectSearchOutput::Ptr>::Ptr& object_search)
        : navigation(navigation), object_search(object_search) {}
  } output_queues_;

  struct Modules {
    NavigationModule::Ptr navigation;
    ObjectSearchModule::Ptr object_search;
    Modules(const NavigationModule::Ptr& navigation,
            const ObjectSearchModule::Ptr& object_search)
        : navigation(navigation), object_search(object_search) {}
  } modules_;

  std::unique_ptr<DsgReceiver> receiver_;

  // These are accessed from multiple threads: make them atomic.
  std::atomic<bool> _find_next{false};
  std::atomic<size_t> _num_objects_to_publish{0};
  std::atomic<bool> _publish_waypoints{false};
  std::atomic<bool> _clear_navigation{false};

  bool _initialized = false;
};

void declare_config(RosNavigationInterface::Config& config);

}  // namespace hydra
