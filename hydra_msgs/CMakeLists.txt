cmake_minimum_required(VERSION 3.1)
project(hydra_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs message_generation semantic_inference_msgs geometry_msgs vision_msgs visualization_msgs)

add_message_files(FILES ActiveLayer.msg
                        ActiveObjectRelationships.msg
                        Detections3D.msg
                        DsgUpdate.msg
                        LabeledRelationships.msg
                        Navigation.msg
                        NavigationPath.msg
                        ObjectSearch.msg
                        ObjectSearchPair.msg
                        VisualRelationshipsEncodings.msg)
add_service_files(FILES GetDsg.srv GetNavigation.srv QueryFreespace.srv Load.srv VLMRelationship.srv)

generate_messages(DEPENDENCIES std_msgs semantic_inference_msgs geometry_msgs vision_msgs visualization_msgs)

catkin_package(
  CATKIN_DEPENDS std_msgs message_runtime
  DEPENDS
  INCLUDE_DIRS
  LIBRARIES
)
