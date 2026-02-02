#!/usr/bin/env python3

import os
import time
from pathlib import Path

import rospy
import roslaunch
import rospkg
from std_msgs.msg import Bool
from hydra_msgs.srv import Save


class RunAll3RScanScenes:
    def __init__(self):
        rospy.init_node("run_all_3rscan_scenes", anonymous=True)

        # Get paths to launch files
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("hydra_ros")
        self.hydra_launch = os.path.join(
            pkg_path, "launch", "datasets", "3rscan.launch"
        )
        self.dataset_launch = os.path.join(
            pkg_path, "launch", "datasets", "publish_3rscan.launch"
        )

        # Load parameters
        self.scenes_paths = rospy.get_param("~scenes_path", "")
        self.output_path = rospy.get_param("~output_path", "")
        self.rate = rospy.get_param("~rate", 1.0)
        self.start_frame = rospy.get_param("~start_frame", 0)
        self.stride = rospy.get_param("~stride", 1)
        self.enable_dsg_lcds = rospy.get_param("~enable_dsg_lcds", False)
        self.use_gt_semantics = rospy.get_param("~use_gt_semantics", False)
        self.open_vocab = rospy.get_param("~open_vocab", False)
        self.segmentation_model = rospy.get_param("~segmentation_model", "yolosam")
        self.use_vlm = rospy.get_param("~use_vlm", False)
        self.min_separation_s = rospy.get_param("~min_separation_s", 0.2)
        self.pub_vlm_annotations = rospy.get_param("~pub_vlm_annotations", False)
        self.start_paused = rospy.get_param("~start_paused", False)
        self.test_path = rospy.get_param("~test_path", "")

        # Validate parameters
        assert (
            self.scenes_paths and Path(self.scenes_paths).exists()
        ), "Invalid or missing '~scenes_path'."
        assert (
            self.output_path and Path(self.output_path).exists()
        ), "Invalid or missing '~output_path'."

        # Get list of scenes to process
        with Path(self.test_path).open("r") as f:
            scenes = [line.strip() for line in f if line.strip()]
        finished_scenes = os.listdir(self.output_path)
        self.scenes = sorted(
            [scene for scene in scenes if scene not in finished_scenes]
        )

        # Finished flag from topic
        self.finished = False
        rospy.Subscriber("finish", Bool, self.callback_finished)

        # Service to save graph
        self.save_graph_service = rospy.ServiceProxy("/hydra_ros_node/save_graph", Save)

    def callback_finished(self, msg):
        self.finished = msg.data

    def run(self):
        if not self.scenes:
            rospy.loginfo("All scenes have already been processed.")
            return

        for scene in self.scenes:
            rospy.loginfo(f"Processing scene: {scene}")
            self.finished = False  # Reset flag
            if not os.path.exists(os.path.join(self.scenes_paths, scene)):
                rospy.logwarn(f"Scene not found: {scene}")
                continue
            rospy.loginfo(f"Starting processing for scene: {scene}")
            # Launch algorithm (hydra)
            uuid1 = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid1)
            dsg_path = Path(self.output_path) / scene
            hydra_args = [
                f"enable_dsg_lcds:={self.enable_dsg_lcds}",
                f"use_gt_semantics:={self.use_gt_semantics}",
                f"open_vocab:={self.open_vocab}",
                f"segmentation_model:={self.segmentation_model}",
                f"use_vlm:={self.use_vlm}",
                f"min_separation_s:={self.min_separation_s}",
                f"pub_vlm_annotations:={self.pub_vlm_annotations}",
                f"dsg_path:={str(dsg_path)}",
            ]
            hydra_launch = roslaunch.parent.ROSLaunchParent(
                uuid1, [(self.hydra_launch, hydra_args)]
            )
            hydra_launch.start()
            rospy.loginfo("Started hydra algorithm.")
            time.sleep(20)  # Give time for nodes to initialize

            # Launch dataset publisher
            uuid2 = roslaunch.rlutil.get_or_generate_uuid(None, False)
            dataset_args = [
                f"dataset_path:={self.scenes_paths}",
                f"rate:={self.rate}",
                f"start_frame:={self.start_frame}",
                f"stride:={self.stride}",
                f"scene_name:={scene}",
                f"start_paused:={self.start_paused}",
            ]
            dataset_launch = roslaunch.parent.ROSLaunchParent(
                uuid2, [(self.dataset_launch, dataset_args)]
            )
            dataset_launch.start()
            rospy.loginfo("Started dataset publisher.")

            # Wait for algorithm to finish
            rospy.loginfo("Waiting for algorithm to finish...")
            rate = rospy.Rate(1)
            while not rospy.is_shutdown() and not self.finished:
                rate.sleep()

            # Save graph
            try:
                rospy.loginfo("Calling save_graph service...")
                self.save_graph_service(True)
                rospy.loginfo("Graph saved.")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to call save_graph service: {e}")

            # Rename output file
            dsg_json_path = dsg_path / "backend" / "dsg_with_mesh.json"
            new_dsg_path = dsg_json_path.parent / "copy_dsg_with_mesh.json"
            if dsg_json_path.exists():
                os.rename(dsg_json_path, new_dsg_path)
                rospy.loginfo("Renamed dsg_with_mesh.json.")
            else:
                rospy.logwarn(f"Expected file not found: {dsg_path}")

            # Shutdown launch files
            hydra_launch.shutdown()
            dataset_launch.shutdown()
            rospy.loginfo(f"Finished processing scene: {scene}")
            time.sleep(20)

        rospy.loginfo("All scenes have been processed.")


if __name__ == "__main__":
    runner = RunAll3RScanScenes()
    runner.run()
