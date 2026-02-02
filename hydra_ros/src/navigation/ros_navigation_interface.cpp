// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/navigation/ros_navigation_interface.h"

namespace hydra {

void declare_config(RosNavigationInterface::Config& conf) {
  using namespace config;
  name("RosNavigationInterfaceConfig");
  field(conf.enable_one_by_one_navigation, "enable_one_by_one_navigation");
  field(conf.always_publish_waypoints, "publish_waypoints");
}

RosNavigationInterface::RosNavigationInterface(
    const Config& config,
    const ros::NodeHandle& nh,
    const ros::NodeHandle& backend_nh,
    const InputQueue<NavigationInput::Ptr>::Ptr& navigation_input_queue,
    const InputQueue<ObjectSearchInput::Ptr>::Ptr& object_search_input_queue,
    const InputQueue<NavigationOutput>::Ptr& navigation_output_queue,
    const InputQueue<ObjectSearchOutput::Ptr>::Ptr& object_search_output_queue,
    const NavigationModule::Ptr& navigation_module,
    const ObjectSearchModule::Ptr object_search_module)
    : node_handlers_(nh, backend_nh),
      input_queues_(navigation_input_queue, object_search_input_queue),
      output_queues_(navigation_output_queue, object_search_output_queue),
      modules_(navigation_module, object_search_module),
      config(config) {
  services_.navigation = node_handlers_.nh.advertiseService(
      "find_paths", &RosNavigationInterface::handleNavigationRequest, this);
  services_.find_next = node_handlers_.nh.advertiseService(
      "find_next", &RosNavigationInterface::handleFindNext, this);
  services_.publish_waypoints = node_handlers_.nh.advertiseService(
      "publish_waypoints", &RosNavigationInterface::handlePublishWaypoints, this);
  services_.clear_navigation = node_handlers_.nh.advertiseService(
      "clear_navigation", &RosNavigationInterface::handleClearNavigation, this);
  // object_search_sub_ = node_handlers_.nh.subscribe(
  //     "object_search", 10, &RosNavigationInterface::callbackObjectSearch, this);
  object_search_sub_ = node_handlers_.nh.subscribe(
      "object_search", 10, &RosNavigationInterface::callbackObjectSearch, this);
  // "/semantic_inference/navigation_prompt_service/navigation_embeddings"
  publishers_.navigation_output =
      node_handlers_.nh.advertise<hydra_msgs::Navigation>("navigation_output", 10);
  publishers_.navigation_goals =
      node_handlers_.nh.advertise<nav_msgs::Path>("navigation_goals", 10);
  publishers_.object_search_output =
      node_handlers_.nh.advertise<hydra_msgs::ObjectSearch>("object_search_output", 10);

  // Dynamic reconfigure server
  dyn_config_server_ =
      std::make_shared<dynamic_reconfigure::Server<hydra_ros::NavigationConfig>>(
          node_handlers_.nh);
  dyn_config_callback_ =
      boost::bind(&RosNavigationInterface::configCallback, this, _1, _2);
  dyn_config_server_->setCallback(dyn_config_callback_);

  // Start the publishing thread
  threads_.spin.reset(new std::thread(&RosNavigationInterface::spin, this));
  threads_.spin_ros.reset(new std::thread(&RosNavigationInterface::spinRos, this));
  // ROS_ERROR("[ros_navigation_interface] Initialized!");
  // Print subscriber topic
  // ROS_INFO("[ros_navigation_interface] Subscribed to object_search topic: %s",
  //          object_search_sub_.getTopic().c_str());
}

bool RosNavigationInterface::handleFindNext(std_srvs::Empty::Request& req,
                                            std_srvs::Empty::Response& res) {
  if (!config.enable_one_by_one_navigation) {
    ROS_WARN("[ros_navigation_interface] One-by-one navigation is disabled!");
    return false;
  }
  if (_num_objects_to_publish.load(std::memory_order_acquire) <= 0) {
    ROS_WARN("[ros_navigation_interface] No more objects to publish!");
    return false;
  }
  _find_next.store(true, std::memory_order_release);
  ROS_INFO("[ros_navigation_interface] One-by-one navigation request received!");
  return true;
}

bool RosNavigationInterface::handlePublishWaypoints(std_srvs::Empty::Request& req,
                                                    std_srvs::Empty::Response& res) {
  if (config.always_publish_waypoints) {
    ROS_WARN(
        "[ros_navigation_interface] Cannot request publish_waypoints when "
        "always_publish_waypoints is enabled!");
    return false;
  }
  if (_num_objects_to_publish.load(std::memory_order_acquire) <= 0) {
    ROS_WARN("[ros_navigation_interface] No more objects to publish!");
    return false;
  }
  _publish_waypoints.store(true, std::memory_order_release);
  ROS_INFO("[ros_navigation_interface] Publish waypoints request received!");
  return true;
}

bool RosNavigationInterface::handleNavigationRequest(
    hydra_msgs::GetNavigation::Request& req, hydra_msgs::GetNavigation::Response& res) {
  NavigationInput::Ptr input = std::make_shared<NavigationInput>();
  input->method = req.method;
  input->explanation = req.explanation;
  if (req.explanation.size() != req.object_ids.size() / 2) {
    res.success = false;
    res.message = "Explanation size does not match object ids!";
    return true;
  }
  if (req.object_ids.size() % 2 != 0) {
    res.success = false;
    res.message = "Object ids must be pairs!";
    return true;
  }
  if (req.object_ids.size() == 0) {
    res.success = false;
    res.message = "No object ids provided!";
    return true;
  }
  for (size_t i = 0; i < req.explanation.size(); ++i) {
    input->object_ids.push_back(
        std::make_pair(req.object_ids[2 * i], req.object_ids[2 * i + 1]));
  }
  res.success = true;
  if (!input_queues_.navigation->push(input)) {
    res.success = false;
    res.message = "Request queue is full!";
  }
  return true;
}

bool RosNavigationInterface::handleClearNavigation(std_srvs::Empty::Request& req,
                                                   std_srvs::Empty::Response& res) {
  if (_num_objects_to_publish.load(std::memory_order_acquire) <= 0) {
    ROS_WARN("[ros_navigation_interface] No objects to clear!");
    return false;
  }
  _clear_navigation.store(true, std::memory_order_release);
  ROS_INFO("[ros_navigation_interface] Clear navigation request received!");
  return true;
}

void RosNavigationInterface::callbackObjectSearch(
    const semantic_inference_msgs::NavigationPromptsEmbeddings::ConstPtr& msg) {
  // ROS_ERROR("[ros_navigation_interface] Received object search request!");
  ObjectSearchInput::Ptr input = std::make_shared<ObjectSearchInput>();
  input->room = msg->room;
  input->prompt = msg->prompt;
  input->object_search = msg->object_search;
  input->text_room_embedding.cols = msg->text_room_embedding.feature.cols;
  input->text_room_embedding.rows = msg->text_room_embedding.feature.rows;
  if (msg->text_room_embedding.feature.data.size() > 0) {
    // Eigen map
    input->text_room_embedding.data = Eigen::Map<Eigen::MatrixXf>(
        const_cast<float*>(msg->text_room_embedding.feature.data.data()),
        msg->text_room_embedding.feature.rows,
        msg->text_room_embedding.feature.cols);
  }
  for (const auto& object_feature : msg->text_object_embedding) {
    ObjectSearchInput::ObjectFeature object;
    object.cols = object_feature.feature.cols;
    object.rows = object_feature.feature.rows;
    if (object_feature.feature.data.size() > 0) {
      // Eigen map
      object.data = Eigen::Map<Eigen::MatrixXf>(
          const_cast<float*>(object_feature.feature.data.data()),
          object_feature.feature.rows,
          object_feature.feature.cols);
    }
    input->text_object_embedding.push_back(object);
  }

  std::unordered_map<std::string, size_t> object_label_index;
  for (size_t i = 0; i < msg->objects.size(); ++i) {
    object_label_index[msg->objects[i]] = i;
  }
  for (const auto& pair : msg->objects_prompt_pairs) {
    ObjectSearchInput::ObjectsPtomptsPair objects_pair;
    objects_pair.prompt = pair.prompt;
    objects_pair.object_label_index = object_label_index[pair.object];
    objects_pair.subject_label_index = object_label_index[pair.subject];
    input->objects_prompt_pairs.push_back(objects_pair);
  }

  if (!input_queues_.object_search->push(input)) {
    ROS_WARN("[ros_navigation_interface] Object search input queue is full!");
  }
}

void RosNavigationInterface::configCallback(hydra_ros::NavigationConfig& cfg,
                                            uint32_t level) {
  if (!_initialized) {
    _initialized = true;
    return;
  }

  config.enable_one_by_one_navigation = cfg.enable_one_by_one_navigation;
  config.always_publish_waypoints = cfg.always_publish_waypoints;
  modules_.object_search->config.room_prob_threshold = cfg.room_prob_threshold;
  modules_.object_search->config.object_prob_threshold = cfg.object_prob_threshold;
  modules_.object_search->config.min_object_vertices = cfg.min_object_vertices;

  // If a disallowed request was already set when toggling config, clear it and warn.
  if (config.always_publish_waypoints &&
      _publish_waypoints.load(std::memory_order_acquire)) {
    _publish_waypoints.store(false, std::memory_order_release);
    ROS_WARN(
        "[ros_navigation_interface] always_publish_waypoints enabled: clearing pending "
        "publish_waypoints request.");
  }
  if (!config.enable_one_by_one_navigation &&
      _find_next.load(std::memory_order_acquire)) {
    _find_next.store(false, std::memory_order_release);
    ROS_WARN(
        "[ros_navigation_interface] disable one-by-one navigation: clearing pending "
        "find_next request.");
  }

  ROS_INFO(
      "[ros_navigation_interface] Dynamic reconfigure called: "
      "enable_one_by_one_navigation = %s, "
      "always_publish_waypoints = %s, "
      "room_prob_threshold = %.3f, object_prob_threshold = %.3f, "
      "min_object_vertices = %zu",
      config.enable_one_by_one_navigation ? "true" : "false",
      config.always_publish_waypoints ? "true" : "false",
      modules_.object_search->config.room_prob_threshold,
      modules_.object_search->config.object_prob_threshold,
      modules_.object_search->config.min_object_vertices);
}

void RosNavigationInterface::publishNavigation(const NavigationOutput& output) {
  hydra_msgs::Navigation nav_msg;
  nav_msg.header.stamp = ros::Time::now();
  nav_msg.header.frame_id = GlobalInfo::instance().getFrames().odom;
  nav_msgs::Path goals_msg;
  goals_msg.header = nav_msg.header;

  const size_t total = output.paths.size();
  _num_objects_to_publish.store(static_cast<int>(total), std::memory_order_release);

  size_t published_count = 0;

  for (size_t idx = 0; idx < total; ++idx) {
    const auto& path = output.paths[idx];

    if (config.enable_one_by_one_navigation) {
      while (!_find_next.load(std::memory_order_acquire) &&
             !_clear_navigation.load(std::memory_order_acquire)) {
        if (!ros::ok()) {
          ROS_WARN(
              "[ros_navigation_interface] ROS shutting down while waiting for "
              "find_next.");
          _num_objects_to_publish.store(0, std::memory_order_release);
          _find_next.store(false, std::memory_order_release);
          return;
        }
        ros::Duration(0.05).sleep();
      }
      _find_next.store(false, std::memory_order_release);
    }
    if (_clear_navigation.load(std::memory_order_acquire)) {
      _num_objects_to_publish.store(0, std::memory_order_release);
      _find_next.store(false, std::memory_order_release);
      _publish_waypoints.store(false, std::memory_order_release);
      _clear_navigation.store(false, std::memory_order_release);
      ROS_INFO("[ros_navigation_interface] Clearing navigation paths.");
      return;
    }

    hydra_msgs::NavigationPath path_msg;
    path_msg.object_id = path.object_id;
    path_msg.target_id = path.target_id;
    path_msg.object_label = path.object_label;
    path_msg.target_label = path.target_label;
    path_msg.explanation = path.explanation;

    for (const auto& point : path.agent_to_target) {
      geometry_msgs::Point point_msg;
      point_msg.x = point.x();
      point_msg.y = point.y();
      point_msg.z = point.z();
      path_msg.agent_to_target.push_back(point_msg);
    }
    for (const auto& point : path.target_to_object) {
      geometry_msgs::Point point_msg;
      point_msg.x = point.x();
      point_msg.y = point.y();
      point_msg.z = point.z();
      path_msg.target_to_object.push_back(point_msg);
    }

    nav_msg.navigation_paths.push_back(path_msg);
    if (!path.agent_to_target.empty()) {
      geometry_msgs::PoseStamped goal1;
      goal1.header = goals_msg.header;
      goal1.pose.position.x = path.agent_to_target.back().x();
      goal1.pose.position.y = path.agent_to_target.back().y();
      goal1.pose.position.z = path.agent_to_target.back().z();
      goal1.pose.orientation.w = 1.0;
      goals_msg.poses.push_back(goal1);
    }
    if (!path.target_to_object.empty()) {
      geometry_msgs::PoseStamped goal2;
      goal2.header = goals_msg.header;
      goal2.pose.position.x = path.target_to_object.back().x();
      goal2.pose.position.y = path.target_to_object.back().y();
      goal2.pose.position.z = path.target_to_object.back().z();
      goal2.pose.orientation.w = 1.0;
      goals_msg.poses.push_back(goal2);
    }

    if (config.enable_one_by_one_navigation) {
      _find_next.store(false, std::memory_order_release);
      publishers_.navigation_output.publish(nav_msg);
      published_count++;
      ROS_INFO("[ros_navigation_interface] Published %zu / %zu navigation paths",
               published_count,
               total);
      bool skip_waypoints = false;

      if (!config.always_publish_waypoints) {
        while (!_publish_waypoints.load(std::memory_order_acquire)) {
          if (!ros::ok()) {
            ROS_WARN(
                "[ros_navigation_interface] ROS shutting down while waiting for "
                "publish_waypoints.");
            _num_objects_to_publish.store(0, std::memory_order_release);
            return;
          }
          if (_find_next.load(std::memory_order_acquire)) {
            skip_waypoints = true;
            break;
          }
          if (_clear_navigation.load(std::memory_order_acquire)) {
            _num_objects_to_publish.store(0, std::memory_order_release);
            _find_next.store(false, std::memory_order_release);
            _publish_waypoints.store(false, std::memory_order_release);
            _clear_navigation.store(false, std::memory_order_release);
            ROS_INFO("[ros_navigation_interface] Clearing navigation paths.");
            return;
          }
          ros::Duration(0.05).sleep();
        }
      }

      if (!goals_msg.poses.empty() && (!skip_waypoints) &&
          (config.always_publish_waypoints ||
           _publish_waypoints.load(std::memory_order_acquire))) {
        publishers_.navigation_goals.publish(goals_msg);
        ROS_INFO("[ros_navigation_interface] Published %zu / %zu navigation goals",
                 published_count,
                 total);
      }

      goals_msg.poses.clear();
      nav_msg.navigation_paths.clear();
      if (!config.always_publish_waypoints) {
        if (!skip_waypoints && _publish_waypoints.load(std::memory_order_acquire)) {
          _publish_waypoints.store(false, std::memory_order_release);
        }
      }
    }
    _num_objects_to_publish.fetch_sub(1, std::memory_order_acq_rel);
  }

  if (!config.enable_one_by_one_navigation) {
    if (!nav_msg.navigation_paths.empty()) {
      publishers_.navigation_output.publish(nav_msg);
    }

    if (!config.always_publish_waypoints) {
      const ros::Time start = ros::Time::now();
      const double timeout = 10.0;
      while (!_publish_waypoints.load(std::memory_order_acquire)) {
        if (!ros::ok()) {
          ROS_WARN(
              "[ros_navigation_interface] ROS shutting down while waiting for "
              "publish_waypoints (non-one-by-one).");
          _num_objects_to_publish.store(0, std::memory_order_release);
          _find_next.store(false, std::memory_order_release);
          return;
        }
        if ((ros::Time::now() - start).toSec() >= timeout) break;
        ros::Duration(0.05).sleep();
      }

      if (!_publish_waypoints.load(std::memory_order_acquire)) {
        ROS_WARN("[ros_navigation_interface] Publish waypoints request timed out!");
        _num_objects_to_publish.store(0, std::memory_order_release);
        _find_next.store(false, std::memory_order_release);
        return;
      }
      _publish_waypoints.store(false, std::memory_order_release);
    }

    if (!goals_msg.poses.empty()) {
      publishers_.navigation_goals.publish(goals_msg);
    }
  }

  _num_objects_to_publish.store(0, std::memory_order_release);
  _find_next.store(false, std::memory_order_release);
  _publish_waypoints.store(false, std::memory_order_release);
}

void RosNavigationInterface::publishObjectSearch(
    const ObjectSearchOutput::Ptr& output) const {
  if (output->object_search) {
    NavigationInput::Ptr input = std::make_shared<NavigationInput>();
    input->method = "dijkstra";
    input->object_search = true;
    for (const auto& object : output->objects) {
      input->object_ids.push_back(std::make_pair(object.id, object.id));
    }
    if (!input_queues_.navigation->push(input)) {
      ROS_WARN("[ros_navigation_interface] Navigation input queue is full!");
    }
    return;
  }
  hydra_msgs::ObjectSearch msg;
  msg.header.stamp = ros::Time::now();
  msg.general_prompt = output->general_prompt;
  for (const auto& object : output->objects) {
    msg.object_ids.push_back(object.id);
    hydra_msgs::ObjectSearchPair object_pair_msg;
    for (size_t i = 0; i < object.relationships.size(); ++i) {
      const auto& feature = object.relationships[i].feature;
      semantic_inference_msgs::FeatureVector feature_msg;
      feature_msg.cols = feature.cols();
      feature_msg.rows = feature.rows();
      for (int r = 0; r < feature.rows(); ++r) {
        for (int c = 0; c < feature.cols(); ++c) {
          feature_msg.data.push_back(feature(r, c));
        }
      }
      object_pair_msg.feature.push_back(feature_msg);
      object_pair_msg.ids.push_back(object.relationships[i].object1);
      object_pair_msg.ids.push_back(object.relationships[i].object2);
      object_pair_msg.labels.push_back(object.relationships[i].object1_label);
      object_pair_msg.labels.push_back(object.relationships[i].object2_label);
      object_pair_msg.prompt.push_back(object.relationships[i].prompt);
      object_pair_msg.num_observations.push_back(
          object.relationships[i].num_observations);
    }
    msg.features.push_back(object_pair_msg);
  }
  publishers_.object_search_output.publish(msg);
}

void RosNavigationInterface::spin() {
  bool should_shutdown = false;
  while (!should_shutdown) {
    bool has_navigation_data = output_queues_.navigation->poll();
    bool has_object_search_data = output_queues_.object_search->poll();
    if (GlobalInfo::instance().force_shutdown() || !has_navigation_data ||
        !has_object_search_data) {
      should_shutdown = should_shutdown_;
    }

    if (!has_navigation_data && !has_object_search_data) {
      continue;
    }
    if (has_navigation_data) {
      publishNavigation(output_queues_.navigation->front());
      output_queues_.navigation->pop();
    }
    if (has_object_search_data) {
      publishObjectSearch(output_queues_.object_search->front());
      output_queues_.object_search->pop();
    }
  }
}

void RosNavigationInterface::spinRos() {
  receiver_.reset(new DsgReceiver(node_handlers_.backend_nh));

  ros::WallRate r(10);
  while (ros::ok()) {
    ros::spinOnce();

    if (receiver_ && receiver_->updated()) {
      if (!receiver_->graph()) {
        r.sleep();
        continue;
      }
      modules_.navigation->setGraph(receiver_->graph());
      modules_.object_search->setGraph(receiver_->graph());
      return;
    }

    r.sleep();
  }
}

}  // namespace hydra
