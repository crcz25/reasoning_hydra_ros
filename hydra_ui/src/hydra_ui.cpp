#include "hydra_ui.h"
namespace hydra_ui {

HydraPanel::HydraPanel(QWidget* parent) : rviz::Panel(parent) {
  hydra_client_find_next_object =
      nh.serviceClient<std_srvs::Empty>("/hydra_ros_node/navigation/find_next");
  hydra_client_publish_waypoints =
      nh.serviceClient<std_srvs::Empty>("/hydra_ros_node/navigation/publish_waypoints");
  hydra_client_clear_navigation =
      nh.serviceClient<std_srvs::Empty>("/hydra_ros_node/navigation/clear_navigation");
  // VLM services
  hydra_client_vlm_process_next = nh.serviceClient<std_srvs::Empty>(
      "/semantic_inference/vlm_for_navigation_node/process_next");
  hydra_client_vlm_publish_current = nh.serviceClient<std_srvs::Empty>(
      "/semantic_inference/vlm_for_navigation_node/publish_current");
  hydra_client_reset_mesh = nh.serviceClient<std_srvs::Empty>(
      "/hydra_dsg_visualizer/reset_object_search_coloring");
  hydra_client_toggle_processing =
      nh.serviceClient<std_srvs::Trigger>("/hydra_ros_node/toggle_image_processing");
  hydra_client_repeat_last_vlm = nh.serviceClient<std_srvs::Empty>(
      "/semantic_inference/vlm_for_navigation_node/repeat_processing");
  hydra_client_stop_vlm = nh.serviceClient<std_srvs::Empty>(
      "/semantic_inference/vlm_for_navigation_node/stop_processing");
  hydra_client_send_task = nh.serviceClient<semantic_inference_msgs::NavigationPrompt>(
      "/semantic_inference/navigation_prompt_service/navigation_prompt");

  QVBoxLayout* v_box_layout = new QVBoxLayout;

  button_find_next_object = new QPushButton;
  button_publish_waypoints = new QPushButton;
  button_clear_navigation = new QPushButton;
  button_vlm_process_next = new QPushButton;
  button_vlm_publish_current = new QPushButton;
  button_reset_mesh = new QPushButton;
  button_toggle_processing = new QPushButton;
  button_repeat_last_vlm = new QPushButton;
  button_stop_vlm = new QPushButton;
  prompt_input = new QLineEdit;
  room_input = new QLineEdit;
  button_send_task = new QPushButton;

  prompt_input->setPlaceholderText("Enter prompt...");
  room_input->setPlaceholderText("Enter room...");
  button_find_next_object->setText("Find Next Object");
  button_publish_waypoints->setText("Publish Waypoints");
  button_clear_navigation->setText("Clear Navigation");
  button_vlm_process_next->setText("VLM Process Next");
  button_vlm_publish_current->setText("VLM Publish Current");
  button_reset_mesh->setText("Reset Mesh");
  button_toggle_processing->setText("Toggle Processing");
  button_repeat_last_vlm->setText("Repeat Last VLM");
  button_stop_vlm->setText("Stop VLM");
  button_send_task->setText("Send Task");

  v_box_layout->addWidget(button_find_next_object);
  v_box_layout->addWidget(button_publish_waypoints);
  v_box_layout->addWidget(button_clear_navigation);
  v_box_layout->addWidget(button_vlm_publish_current);
  v_box_layout->addWidget(button_vlm_process_next);
  v_box_layout->addWidget(button_repeat_last_vlm);
  v_box_layout->addWidget(button_stop_vlm);
  v_box_layout->addWidget(button_reset_mesh);
  v_box_layout->addWidget(button_toggle_processing);
  v_box_layout->addWidget(button_send_task);
  v_box_layout->addWidget(new QLabel("Task:"));
  v_box_layout->addWidget(prompt_input);
  v_box_layout->addWidget(new QLabel("Room:"));
  v_box_layout->addWidget(room_input);

  QVBoxLayout* global_vbox_layout = new QVBoxLayout;
  QHBoxLayout* global_hbox_layout = new QHBoxLayout;

  global_vbox_layout->addLayout(global_hbox_layout);
  v_box_layout->addLayout(global_vbox_layout);

  setLayout(v_box_layout);

  connect(
      button_find_next_object, SIGNAL(clicked()), this, SLOT(onFindNextObjectClick()));
  connect(button_publish_waypoints,
          SIGNAL(clicked()),
          this,
          SLOT(onPublishWaypointsClick()));
  connect(
      button_clear_navigation, SIGNAL(clicked()), this, SLOT(onClearNavigationClick()));
  connect(
      button_vlm_process_next, SIGNAL(clicked()), this, SLOT(onVLMProcessNextClick()));
  connect(button_vlm_publish_current,
          SIGNAL(clicked()),
          this,
          SLOT(onVLMPublishCurrentClick()));
  connect(button_reset_mesh, SIGNAL(clicked()), this, SLOT(onResetMeshClick()));
  connect(button_toggle_processing,
          SIGNAL(clicked()),
          this,
          SLOT(onToggleProcessingClick()));
  connect(
      button_repeat_last_vlm, SIGNAL(clicked()), this, SLOT(onRepeatLastVLMClick()));
  connect(button_stop_vlm, SIGNAL(clicked()), this, SLOT(onStopVLMClick()));
  connect(button_send_task, SIGNAL(clicked()), this, SLOT(onSendTaskClick()));
}

void HydraPanel::onFindNextObjectClick() {
  std_srvs::Empty srv;
  if (!hydra_client_find_next_object.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_find_next_object.getService().c_str());
  }
}

void HydraPanel::onPublishWaypointsClick() {
  std_srvs::Empty srv;
  if (!hydra_client_publish_waypoints.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_publish_waypoints.getService().c_str());
  }
}

void HydraPanel::onClearNavigationClick() {
  std_srvs::Empty srv;
  if (!hydra_client_clear_navigation.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_clear_navigation.getService().c_str());
  }
}

void HydraPanel::onVLMProcessNextClick() {
  std_srvs::Empty srv;
  if (!hydra_client_vlm_process_next.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_vlm_process_next.getService().c_str());
  }
}

void HydraPanel::onVLMPublishCurrentClick() {
  std_srvs::Empty srv;
  if (!hydra_client_vlm_publish_current.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_vlm_publish_current.getService().c_str());
  }
}

void HydraPanel::onResetMeshClick() {
  std_srvs::Empty srv;
  if (!hydra_client_reset_mesh.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_reset_mesh.getService().c_str());
  }
}

void HydraPanel::onToggleProcessingClick() {
  std_srvs::Trigger srv;
  if (!hydra_client_toggle_processing.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_toggle_processing.getService().c_str());
  } else {
    if (srv.response.success) {
      ROS_INFO("[HYDRA-UI] Image processing toggled successfully.");
    } else {
      ROS_WARN("[HYDRA-UI] Image processing toggle failed: %s",
               srv.response.message.c_str());
    }
  }
}

void HydraPanel::onRepeatLastVLMClick() {
  std_srvs::Empty srv;
  if (!hydra_client_repeat_last_vlm.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_repeat_last_vlm.getService().c_str());
  }
}

void HydraPanel::onStopVLMClick() {
  std_srvs::Empty srv;
  if (!hydra_client_stop_vlm.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_stop_vlm.getService().c_str());
  }
}

void HydraPanel::onSendTaskClick() {
  semantic_inference_msgs::NavigationPrompt srv;
  srv.request.prompt = prompt_input->text().toStdString();
  srv.request.room = room_input->text().toStdString();

  if (srv.request.prompt.empty() || srv.request.room.empty()) {
    ROS_ERROR("[HYDRA-UI] Prompt and room must not be empty.");
    return;
  }
  if (!hydra_client_send_task.call(srv)) {
    ROS_ERROR("[HYDRA-UI] Service call failed: %s",
              hydra_client_send_task.getService().c_str());
  } else {
    if (!srv.response.error.empty()) {
      ROS_ERROR("[HYDRA-UI] Error from service: %s", srv.response.error.c_str());
    } else {
      // Log output from the service
      std::string output = "[HYDRA-UI] Task sent successfully.\n";
      output += "  Objects to find: ";
      for (const auto& object : srv.response.response.objects) {
        output += object + " ";
      }
      output +=
          "\n  Object search: " + std::to_string(srv.response.response.object_search) +
          "\n";
      if (!srv.response.response.objects_prompt_pairs.empty()) {
        output += "  Object pairs and prompts:\n";
        for (const auto& pair : srv.response.response.objects_prompt_pairs) {
          output += "    Object: " + pair.object + ", Subject: " + pair.subject + "\n";
          output += "      Prompt: " + pair.prompt + "\n";
        }
      }
      ROS_INFO("%s", output.c_str());
    }
  }
}

void HydraPanel::save(rviz::Config config) const { rviz::Panel::save(config); }
void HydraPanel::load(const rviz::Config& config) { rviz::Panel::load(config); }

}  // namespace hydra_ui

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(hydra_ui::HydraPanel, rviz::Panel)
