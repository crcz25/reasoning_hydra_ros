#pragma once

#include <ros/ros.h>
#include <semantic_inference_msgs/NavigationPrompt.h>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
#include <stdio.h>

#include <string>

#ifndef Q_MOC_RUN
#include <rviz/panel.h>

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>
#endif

class QLineEdit;
class QPushButton;

namespace hydra_ui {
class HydraPanel : public rviz::Panel {
  Q_OBJECT

 public:
  explicit HydraPanel(QWidget* parent = 0);
  virtual void load(const rviz::Config& config);
  virtual void save(rviz::Config config) const;

 public Q_SLOTS:
  void onFindNextObjectClick();
  void onPublishWaypointsClick();
  void onClearNavigationClick();
  void onVLMProcessNextClick();
  void onVLMPublishCurrentClick();
  void onResetMeshClick();
  void onToggleProcessingClick();
  void onRepeatLastVLMClick();
  void onStopVLMClick();
  void onSendTaskClick();
 protected Q_SLOTS:

 protected:
  QPushButton* button_find_next_object;
  ros::ServiceClient hydra_client_find_next_object;

  QPushButton* button_publish_waypoints;
  ros::ServiceClient hydra_client_publish_waypoints;

  QPushButton* button_clear_navigation;
  ros::ServiceClient hydra_client_clear_navigation;

  QPushButton* button_vlm_process_next;
  ros::ServiceClient hydra_client_vlm_process_next;

  QPushButton* button_vlm_publish_current;
  ros::ServiceClient hydra_client_vlm_publish_current;

  QPushButton* button_reset_mesh;
  ros::ServiceClient hydra_client_reset_mesh;

  QPushButton* button_toggle_processing;
  ros::ServiceClient hydra_client_toggle_processing;

  QPushButton* button_repeat_last_vlm;
  ros::ServiceClient hydra_client_repeat_last_vlm;

  QPushButton* button_stop_vlm;
  ros::ServiceClient hydra_client_stop_vlm;

  QLineEdit* prompt_input;
  QLineEdit* room_input;
  QPushButton* button_send_task;
  ros::ServiceClient hydra_client_send_task;

  ros::NodeHandle nh;
};

}  // namespace hydra_ui
