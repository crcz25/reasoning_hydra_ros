// Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and
// Technology All rights reserved.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "hydra_ros/hydra_vis_pipeline.h"

namespace hydra {

HydraVisPipeline::HydraVisPipeline(const ros::NodeHandle& nh, int robot_id)
    : HydraPipeline(config::fromRos<PipelineConfig>(nh), robot_id),
      config_(config::checkValid(config::fromRos<HydraRosConfig>(nh))),
      nh_(nh) {}

HydraVisPipeline::~HydraVisPipeline() {}

void HydraVisPipeline::init() {
  const auto& pipeline_config = GlobalInfo::instance().getConfig();
  initFrontend();
  ros::NodeHandle bnh(nh_, "backend");
  initBackend(bnh);
  initReconstruction();
  initNavigation(bnh);
}

void HydraVisPipeline::initFrontend() {
  ros::NodeHandle fnh(nh_, "frontend");
  const auto logs = GlobalInfo::instance().getLogs();
  FrontendModule::Ptr frontend =
      config::createFromROS<FrontendModule>(fnh, frontend_dsg_, shared_state_, logs);
  if (config_.enable_frontend_output) {
    CHECK(frontend) << "Frontend module required!";
    frontend->addSink(std::make_shared<RosFrontendPublisher>(
        fnh, config::fromRos<RosFrontendPublisher::Config>(fnh)));
  }

  modules_["frontend"] = frontend;
}

void HydraVisPipeline::initBackend(const ros::NodeHandle& bnh) {
  const auto logs = GlobalInfo::instance().getLogs();
  BackendModule::Ptr backend = config::createFromROS<BackendModule>(
      bnh, backend_dsg_, shared_state_, GlobalInfo::instance().getLogs());
  CHECK(backend) << "Failed to construct backend!";
  backend->addSink(std::make_shared<RosBackendPublisher>(bnh));

  if (backend->config.use_vlm) {
    ros::NodeHandle blnh(bnh, "vlm_relationships");
    shared_state_->vlm_labels_queue.reset(new InputQueue<BackendVLMLabelsInput::Ptr>());
    vlm_relationships_ = std::make_shared<RosVLMRelationships>(
        blnh,
        config::checkValid(config::fromRos<RosVLMRelationships::Config>(blnh)),
        shared_state_->vlm_labels_queue);
    backend->addSink(vlm_relationships_);
  }
  modules_["backend"] = backend;

  const auto frontend = getModule<FrontendModule>("frontend");
  if (!frontend) {
    LOG(ERROR) << "Invalid frontend module! Not setting 2D places update";
    return;
  }

  if (frontend->config.surface_places) {
    auto places_functor =
        std::make_shared<Update2dPlacesFunctor>(backend->config.places2d_config);
    backend->setUpdateFunctor(DsgLayers::MESH_PLACES, places_functor);
  }

  if (frontend->config.use_frontiers && frontend->config.frontier_places) {
    auto frontiers_functor =
        std::make_shared<UpdateFrontiersFunctor>(backend->config.frontier_config);
    backend->setUpdateFunctor(DsgLayers::BUILDINGS + 1, frontiers_functor);
  }
}

void HydraVisPipeline::initReconstruction() {
  const auto frontend = getModule<FrontendModule>("frontend");
  if (!frontend) {
    LOG(ERROR) << "Invalid frontend module: disabling reconstruction";
    return;
  }

  ros::NodeHandle rnh(nh_, "reconstruction");
  modules_["reconstruction"] =
      config::createFromROS<ReconstructionModule>(rnh, frontend->getQueue());
}

void HydraVisPipeline::initLCD() {}

void HydraVisPipeline::initNavigation(const ros::NodeHandle& bnh) {
  ros::NodeHandle nnh(nh_, "navigation");
  ros::NodeHandle osnh(nh_, "object_search");
  NavigationModule::Ptr navigation_module =
      std::make_shared<NavigationModule>(NavigationModule::Config{});
  ObjectSearchModule::Ptr object_search_module =
      config::createFromROS<ObjectSearchModule>(osnh);
  RosNavigationInterface::Config navigation_config =
      config::fromRos<RosNavigationInterface::Config>(osnh);
  CHECK(navigation_module) << "Failed to construct navigation module!";
  navigation_interface_ =
      std::make_unique<RosNavigationInterface>(navigation_config,
                                               nnh,
                                               bnh,
                                               navigation_module->inputQueue(),
                                               object_search_module->inputQueue(),
                                               navigation_module->outputQueue(),
                                               object_search_module->outputQueue(),
                                               navigation_module,
                                               object_search_module);
  modules_["navigation"] = navigation_module;
  modules_["object_search"] = object_search_module;
}

void HydraVisPipeline::publishVisualization() {
  const auto backend = getModule<BackendModule>("backend");
  const auto frontend = getModule<FrontendModule>("frontend");
  if (!frontend) {
    LOG(ERROR) << "Invalid frontend module!";
    return;
  }
  if (!backend) {
    LOG(ERROR) << "Invalid backend module!";
    return;
  }
  // Publish the visualization data
  frontend->callSinks();
  backend->callSinks();
}

void HydraVisPipeline::loadGraph(const std::string& filepath) {
  auto graph = hydra::DynamicSceneGraph::load(filepath);
  auto backend = getModule<BackendModule>("backend");
  auto frontend = getModule<FrontendModule>("frontend");
  if (!graph || !backend || !frontend) {
    LOG(ERROR) << "Failed to load graph or invalid modules!";
    return;
  }
  backend->setGraph(graph);
  frontend->setGraph(graph->clone());
}

}  // namespace hydra
