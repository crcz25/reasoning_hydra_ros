#!/usr/bin/env python3
import rospy
import threading
import sys
import select
from pathlib import Path
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from PIL import Image as PILImage
import numpy as np
import json
import cv_bridge
import tf
import termios
import tty
import time
import cv2
from tqdm import trange


class ReplicaPublisher:
    def __init__(self):
        self.replica_path = Path(rospy.get_param("~dataset_path", ""))
        self.cache = rospy.get_param("~cache", False)
        self.rate = rospy.get_param("~rate", 10)
        self.start_paused = rospy.get_param("~start_paused", True)
        self.paused = self.start_paused
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.odom_frame = rospy.get_param("~odom_frame", "world")
        self.sensor_frame = rospy.get_param("~sensor_frame", "sensor")
        self.scaling_factor = rospy.get_param("~scale", 1.0)
        print(f"Scaling factor: {self.scaling_factor}")
        self.crop_width = rospy.get_param("~crop_width", 0)
        self.crop_height = rospy.get_param("~crop_height", 0)
        self.start_frame = rospy.get_param("~start_frame", 0)
        self.stride = rospy.get_param("~stride", 1)

        (
            self.images,
            self.depth,
            self.poses,
            self.K,
            self.scale,
            self.W,
            self.H,
        ) = self.load_replica()

        self.camera_info_pub = rospy.Publisher(
            "rgb_camera_info", CameraInfo, queue_size=1
        )
        self.image_pub = rospy.Publisher("rgb_image", Image, queue_size=1)
        self.depth_info_pub = rospy.Publisher(
            "depth_camera_info", CameraInfo, queue_size=1
        )
        self.depth_pub = rospy.Publisher("depth_image", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.cv_bridge = cv_bridge.CvBridge()

        self.key_listener_thread = threading.Thread(
            target=self.listen_for_spacebar, daemon=True
        )
        self.key_listener_thread.start()
        rospy.loginfo("[ReplicaPublisher] Initialized.")

    def load_replica(self):
        images, depth, poses, K = [], [], [], []
        with open(self.replica_path.parent / "cam_params.json") as file:
            data = json.load(file)
            camera_params = data.get("camera", {})
            w, h = camera_params.get("w"), camera_params.get("h")
            crop_height, crop_width = 0, 0
            if self.crop_width > 0 and self.crop_width < w:
                crop_width = (w - self.crop_width) // 2
                w = self.crop_width
            if self.crop_height > 0 and self.crop_height < h:
                h = self.crop_height
                crop_height = (h - self.crop_height) // 2
            fx, fy = camera_params.get("fx"), camera_params.get("fy")
            cx, cy = camera_params.get("cx"), camera_params.get("cy")
            scale = camera_params.get("scale", 1.0)
            # Adjust the intrinsics for the crop
            cx -= crop_width
            cy -= crop_height
            # Scale the intrinsics
            fx, fy, cx, cy = (
                fx * self.scaling_factor,
                fy * self.scaling_factor,
                cx * self.scaling_factor,
                cy * self.scaling_factor,
            )
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Get number of files in results directory
        num_files = len(list((self.replica_path / "results").glob("*.png")))
        for i in range(0, num_files, self.stride):
            file = self.replica_path / "results" / f"depth{i:06d}.png"
            depth.append(
                np.asarray(PILImage.open(file)) / scale if self.cache else file
            )
            file = self.replica_path / "results" / f"frame{i:06d}.jpg"
            images.append(np.asarray(PILImage.open(file)) if self.cache else file)

        with open(self.replica_path / "traj.txt") as file:
            for i, line in enumerate(file):
                if i % self.stride != 0:
                    continue
                poses.append(
                    np.array([float(val) for val in line.split()]).reshape((4, 4))
                )

        return images, depth, poses, K, scale, w, h

    def listen_for_spacebar(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not rospy.is_shutdown():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == " ":
                        self.paused = not self.paused
                        rospy.loginfo(
                            "Publishing {}".format(
                                "resumed" if not self.paused else "paused"
                            )
                        )
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def publish_replica(self):
        for i in trange(
            self.start_frame, len(self.images), desc="Publishing", unit="frame"
        ):
            pose = self.poses[i]
            image = self.images[i]
            depth = self.depth[i]
            while self.paused and not rospy.is_shutdown():
                time.sleep(0.1)

            image_stamp = Header()
            image_stamp.frame_id = self.sensor_frame
            image_stamp.stamp = rospy.Time.from_sec(time.time())

            if not self.cache:
                image = np.asarray(PILImage.open(image))
                depth = np.asarray(PILImage.open(depth)) / self.scale
            # Crop then resize
            if self.crop_width > 0 and self.crop_height > 0:
                crop_width = (image.shape[1] - self.crop_width) // 2
                crop_height = (image.shape[0] - self.crop_height) // 2
                image = image[crop_height:-crop_height, crop_width:-crop_width]
                depth = depth[crop_height:-crop_height, crop_width:-crop_width]

            if self.scale != 1.0:
                image = cv2.resize(
                    image,
                    (
                        int(self.W * self.scaling_factor),
                        int(self.H * self.scaling_factor),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                depth = cv2.resize(
                    depth,
                    (
                        int(self.W * self.scaling_factor),
                        int(self.H * self.scaling_factor),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )

            self.publish_image(image, image_stamp)
            self.publish_depth(depth, image_stamp)
            self.publish_camera_info(image_stamp)
            self.publish_depth_info(image_stamp)
            self.publish_pointcloud(image_stamp, depth, image, pose)
            self.publish_tf(pose, image_stamp.stamp)
            time.sleep(1 / self.rate)
        rospy.loginfo("Finished publishing Replica data.")

    def publish_image(self, image, header):
        self.image_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(image, encoding="rgb8", header=header)
        )

    def publish_depth(self, depth, header):
        # Convert to 32FC1
        depth = depth.astype(np.float32)
        self.depth_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(depth, encoding="32FC1", header=header)
        )

    def publish_camera_info(self, header):
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width, camera_info.height = self.W, self.H
        camera_info.K = self.K.flatten()
        camera_info.P = [
            self.K[0, 0],
            0,
            self.K[0, 2],
            0,
            0,
            self.K[1, 1],
            self.K[1, 2],
            0,
            0,
            0,
            1,
            0,
        ]
        self.camera_info_pub.publish(camera_info)

    def publish_depth_info(self, header):
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width, camera_info.height = self.W, self.H
        camera_info.K = self.K.flatten()
        camera_info.P = [
            self.K[0, 0],
            0,
            self.K[0, 2],
            0,
            0,
            self.K[1, 1],
            self.K[1, 2],
            0,
            0,
            0,
            1,
            0,
        ]
        self.depth_info_pub.publish(camera_info)

    def publish_tf(self, pose, stamp):
        translation = pose[:3, 3]
        rotation = tf.transformations.quaternion_from_matrix(pose)
        self.tf_broadcaster.sendTransform(
            translation, rotation, stamp, self.robot_frame, self.odom_frame
        )
        self.tf_broadcaster.sendTransform(
            (0, 0, 0), (0, 0, 0, 1), stamp, self.sensor_frame, self.robot_frame
        )

    def publish_pointcloud(self, header, depth_image, rgb_image, pose):
        header.frame_id = self.odom_frame
        height, width = depth_image.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u, v = u.flatten(), v.flatten()
        depth = depth_image.flatten()

        # Filter out points with zero depth
        mask = depth > 0
        u, v, depth = u[mask], v[mask], depth[mask]

        # Convert pixel coordinates to camera frame (3D points)
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points_camera = np.vstack((x, y, z, np.ones_like(x)))  # Homogeneous coordinates

        # Transform points to world coordinates
        points_world = pose @ points_camera
        x_w, y_w, z_w = points_world[:3, :]

        # Get corresponding RGB values
        rgb_values = rgb_image[v, u]  # Extract RGB using the valid mask
        rgb_packed = (
            (rgb_values[:, 0].astype(np.uint32) << 16)
            | (rgb_values[:, 1].astype(np.uint32) << 8)
            | rgb_values[:, 2].astype(np.uint32)
        )

        # Create PointCloud2 message
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        points = list(zip(x_w, y_w, z_w, rgb_packed))

        pc2_msg = pc2.create_cloud(header, fields, points)
        self.pointcloud_pub.publish(pc2_msg)


def main():
    rospy.init_node("publish_replica", anonymous=True)
    publisher = ReplicaPublisher()
    rospy.loginfo("Publishing replica... Press space to pause/resume.")
    publisher.publish_replica()


if __name__ == "__main__":
    main()
