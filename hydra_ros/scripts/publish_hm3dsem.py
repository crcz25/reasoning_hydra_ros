#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import math
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from PIL import Image as PILImage
import cv_bridge
import tf
import termios
import tty
import rospy
import threading
import os
import sys
import select
import time
import cv2
from tqdm import trange


class HM3DSemPublisher:
    def __init__(self):
        self.dataset_path = Path(rospy.get_param("~dataset_path", ""))
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

        self.camera_info_pub = rospy.Publisher(
            "rgb_camera_info", CameraInfo, queue_size=1
        )
        self.image_pub = rospy.Publisher("rgb_image", Image, queue_size=1)
        self.depth_info_pub = rospy.Publisher(
            "depth_camera_info", CameraInfo, queue_size=1
        )
        self.depth_pub = rospy.Publisher("depth_image", Image, queue_size=1)
        self.pointcloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=1)
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.data_list = self._get_data_list()
        self.H = self._load_image(self.data_list[0][0]).shape[0]
        self.W = self._load_image(self.data_list[0][0]).shape[1]
        self.depth_intrinsics, self.H, self.W = self._load_depth_intrinsics(
            self.H, self.W
        )
        self.scale = 1000.0

        self.cv_bridge = cv_bridge.CvBridge()

        self.key_listener_thread = threading.Thread(
            target=self.listen_for_spacebar, daemon=True
        )
        self.key_listener_thread.start()

    def _load_depth_intrinsics(self, H, W):
        """
        Load the depth camera intrinsics.

        Returns:
            Depth camera intrinsics as a numpy array (3x3 matrix).
        """
        hfov = 90 * np.pi / 180
        vfov = 2 * math.atan(np.tan(hfov / 2) * H / W)
        fx = W / (2.0 * np.tan(hfov / 2.0))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        cx = W / 2
        cy = H / 2
        crop_height, crop_width = 0, 0
        if self.crop_width > 0 and self.crop_width < W:
            crop_width = (W - self.crop_width) // 2
            W = self.crop_width
        if self.crop_height > 0 and self.crop_height < H:
            H = self.crop_height
            crop_height = (H - self.crop_height) // 2
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
        depth_camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return depth_camera_matrix, H, W

    def _get_data_list(self):
        """
        Get a list of RGB-D data samples based on the dataset format and mode.

        Returns:
            List of RGB-D data samples (RGB image path, depth image path).
        """
        rgb_data_list = []
        depth_data_list = []
        pose_data_list = []
        rgb_data_list = os.listdir(self.dataset_path / "rgb")
        rgb_data_list = [self.dataset_path / "rgb" / x for x in rgb_data_list]
        depth_data_list = os.listdir(self.dataset_path / "depth")
        depth_data_list = [self.dataset_path / "depth" / x for x in depth_data_list]
        pose_data_list = os.listdir(self.dataset_path / "pose")
        pose_data_list = [self.dataset_path / "pose" / x for x in pose_data_list]
        # sort the data list
        rgb_data_list.sort()
        depth_data_list.sort()
        pose_data_list.sort()
        return list(zip(rgb_data_list, depth_data_list, pose_data_list))

    def _load_image(self, path):
        """
        Load the RGB image from the given path.

        Args:
            path: Path to the RGB image file.

        Returns:
            RGB image as a numpy array.
        """
        # Load the RGB image using PIL
        rgb_image = PILImage.open(path)
        return np.asarray(rgb_image)

    def _load_depth(self, path):
        """
        Load the depth image from the given path.

        Args:
            path: Path to the depth image file.

        Returns:
            Depth image as a numpy array.
        """
        # Load the depth image using OpenCV
        depth_image = PILImage.open(path)
        return np.asarray(depth_image) / self.scale

    def _load_pose(self, path):
        """
        Load the camera pose from the given path.

        Args:
            path: Path to the camera pose file.

        Returns:
            Camera pose as a numpy array (4x4 matrix).
        """
        with open(path, "r") as file:
            line = file.readline().strip()
            values = line.split()
            values = [float(val) for val in values]
            transformation_matrix = np.array(values).reshape((4, 4))
            C = np.eye(4)
            C[1, 1] = -1
            C[2, 2] = -1
            transformation_matrix = np.matmul(transformation_matrix, C)
            C = np.zeros_like(C)
            C[0, 2] = 1
            C[1, 1] = 1
            C[2, 0] = -1
            C[3, 3] = 1
            transformation_matrix = np.matmul(C, transformation_matrix)
            C = np.zeros_like(C)
            C[0, 0] = 1
            C[1, 2] = -1
            C[2, 1] = 1
            C[3, 3] = 1
            transformation_matrix = np.matmul(C, transformation_matrix)
        return transformation_matrix

    def _getitem(self, idx):
        """
        Get a data sample based on the given index.

        Args:
            idx: Index of the data sample.

        Returns:
            RGB image and depth image as numpy arrays.
        """
        rgb_path, depth_path, pose_path = self.data_list[idx]
        rgb_image = self._load_image(rgb_path)
        depth_image = self._load_depth(depth_path)
        pose = self._load_pose(pose_path)
        return rgb_image, depth_image, pose

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

    def publish_hm3dsem(self):
        for idx in trange(
            self.start_frame,
            len(self.data_list),
            self.stride,
            desc="Publishing",
            unit="frame",
        ):
            while self.paused and not rospy.is_shutdown():
                time.sleep(0.1)
            rgb_image, depth_image, pose = self._getitem(idx)
            # Crop then resize
            if self.crop_width > 0 and self.crop_height > 0:
                crop_width = (rgb_image.shape[1] - self.crop_width) // 2
                crop_height = (rgb_image.shape[0] - self.crop_height) // 2
                rgb_image = rgb_image[crop_height:-crop_height, crop_width:-crop_width]
                depth_image = depth_image[
                    crop_height:-crop_height, crop_width:-crop_width
                ]

            if self.scale != 1.0:
                rgb_image = cv2.resize(
                    rgb_image,
                    (
                        int(self.W * self.scaling_factor),
                        int(self.H * self.scaling_factor),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                depth_image = cv2.resize(
                    depth_image,
                    (
                        int(self.W * self.scaling_factor),
                        int(self.H * self.scaling_factor),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
            image_stamp = Header()
            image_stamp.frame_id = self.sensor_frame
            image_stamp.stamp = rospy.Time.now()
            self.publish_tf(pose, image_stamp.stamp)
            self.publish_image(rgb_image, image_stamp)
            self.publish_depth(depth_image, image_stamp)
            self.publish_camera_info(image_stamp)
            self.publish_depth_info(image_stamp)
            self.publish_pointcloud(image_stamp, depth_image, rgb_image, pose)
            time.sleep(1 / self.rate)
        rospy.loginfo("Finished publishing HM3dSem data.")

    def publish_image(self, image, header):
        self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(image, "rgb8", header))

    def publish_depth(self, depth, header):
        self.depth_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(depth.astype(np.float32), "32FC1", header)
        )

    def publish_camera_info(self, header):
        camera_info = CameraInfo()
        camera_info.header = header
        camera_info.width, camera_info.height = self.W, self.H
        camera_info.K = self.depth_intrinsics.flatten()
        camera_info.P = [
            self.depth_intrinsics[0, 0],
            0,
            self.depth_intrinsics[0, 2],
            0,
            0,
            self.depth_intrinsics[1, 1],
            self.depth_intrinsics[1, 2],
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
        camera_info.K = self.depth_intrinsics.flatten()
        camera_info.P = [
            self.depth_intrinsics[0, 0],
            0,
            self.depth_intrinsics[0, 2],
            0,
            0,
            self.depth_intrinsics[1, 1],
            self.depth_intrinsics[1, 2],
            0,
            0,
            0,
            1,
            0,
        ]
        self.depth_info_pub.publish(camera_info)

    def publish_pointcloud(self, header, depth_image, rgb_image, pose):
        header.frame_id = self.odom_frame
        height, width = depth_image.shape
        fx, fy = self.depth_intrinsics[0, 0], self.depth_intrinsics[1, 1]
        cx, cy = self.depth_intrinsics[0, 2], self.depth_intrinsics[1, 2]

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

    def publish_tf(self, pose, stamp):
        translation = pose[:3, 3]
        rotation = tf.transformations.quaternion_from_matrix(pose)
        self.tf_broadcaster.sendTransform(
            translation, rotation, stamp, self.robot_frame, self.odom_frame
        )
        self.tf_broadcaster.sendTransform(
            (0, 0, 0), (0, 0, 0, 1), stamp, self.sensor_frame, self.robot_frame
        )

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.robot_frame
        msg.pose.pose.position.x = translation[0]
        msg.pose.pose.position.y = translation[1]
        msg.pose.pose.position.z = translation[2]
        msg.pose.pose.orientation.x = rotation[0]
        msg.pose.pose.orientation.y = rotation[1]
        msg.pose.pose.orientation.z = rotation[2]
        msg.pose.pose.orientation.w = rotation[3]

        self.odom_pub.publish(msg)


def main():
    rospy.init_node("publish_hm3dsem")
    hm3dsem_publisher = HM3DSemPublisher()
    rospy.loginfo("Publishing HM3DSEM data... Press space to pause/resume.")
    hm3dsem_publisher.publish_hm3dsem()
    rospy.spin()


if __name__ == "__main__":
    main()
