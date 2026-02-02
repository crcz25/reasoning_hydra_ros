#!/usr/bin/env python3

import rospy
import cv_bridge
from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
import cv2
import open3d as o3d
import threading
from std_msgs.msg import String
from pathlib import Path

import tf.transformations as tft


class D455Project:
    def __init__(self):
        self.pcl_sub = rospy.Subscriber(
            "/os_cloud_node/points", PointCloud2, self.callback_cloud
        )
        self.color_sub = rospy.Subscriber(
            "/cam0/cam0/compressed", CompressedImage, self.callback_image
        )

        self.signal_sub = rospy.Subscriber("/signal", String, self.callback_signal)
        self.point_cloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=10)
        self.accumulated_points = o3d.geometry.PointCloud()
        self.colored_points = o3d.geometry.PointCloud()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = cv_bridge.CvBridge()

        self.camera_frame = rospy.get_param("~sensor_frame", "cam0")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.min_range = rospy.get_param("~min_range", 0.1)
        self.max_range = rospy.get_param("~max_range", 5.0)
        self.voxel_size = rospy.get_param("~voxel_size", 0.05)

        self.cam_projection_matrix = np.array(
            [
                545.274273,
                0.000000,
                370.473852,
                0.0,
                0.000000,
                545.428903,
                266.419781,
                0.0,
                0.000000,
                0.000000,
                1.000000,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ).reshape(4, 4)
        self.lock = threading.Lock()

        rospy.loginfo("RMF project node initialized.")

    def transform_to_matrix(self, transform: TransformStamped) -> np.ndarray:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        trans = tft.translation_matrix([translation.x, translation.y, translation.z])
        rot = tft.quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        return np.dot(trans, rot)

    def callback_cloud(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                msg.header.frame_id,
                msg.header.stamp,
                rospy.Duration(1.0),
            )

            # Convert TF transform to a 4x4 matrix
            T = self.transform_to_matrix(transform)

            # Extract point cloud
            points = list(
                pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
            )
            if not points:
                return

            np_points = np.array(points)

            # Add homogeneous coordinate (Nx4) to apply 4x4 transform
            ones = np.ones((np_points.shape[0], 1))
            points_hom = np.hstack((np_points, ones))  # Shape: (N, 4)

            # Transform to world frame
            transformed_points = (T @ points_hom.T).T[:, :3]  # Back to (N, 3)

            # Accumulate into Open3D
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(transformed_points)
            with self.lock:
                self.accumulated_points += cloud
                self.accumulated_points = self.accumulated_points.voxel_down_sample(
                    voxel_size=self.voxel_size
                )

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Failed to get transform for point cloud.")

    def callback_image(self, color_msg: CompressedImage):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                color_msg.header.stamp,
                rospy.Duration(1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Failed to get camera pose in world frame.")
            return
        rospy.loginfo("Color image.")
        with self.lock:
            if len(self.accumulated_points.points) == 0:
                rospy.logwarn("No point cloud data to publish.")
                return
            # Pose in homogeneous transformation matrix.
            pose = np.eye(4)
            pose[:3, 3] = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
            q = transform.transform.rotation
            # Convert quaternion to rotation matrix.
            q = [q.x, q.y, q.z, q.w]
            R = tft.quaternion_matrix(q)[:3, :3]
            pose[:3, :3] = R
            # Get the color image.
            color_image = self.cv_bridge.compressed_imgmsg_to_cv2(
                color_msg, "passthrough"
            )
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            color_image = np.array(color_image, dtype=np.uint8)
            # Get the camera intrinsics.

            self.publish_pointcloud(color_msg.header, color_image, np.linalg.inv(pose))

    def publish_pointcloud(self, header, rgb_image, pose):
        header.frame_id = self.world_frame

        if len(self.accumulated_points.points) == 0:
            rospy.logwarn("No accumulated points to publish")
            return

        height, width = rgb_image.shape[:2]
        fx, fy = self.cam_projection_matrix[0, 1], self.cam_projection_matrix[1, 1]
        cx, cy = self.cam_projection_matrix[0, 2], self.cam_projection_matrix[1, 2]

        # Get accumulated point cloud as numpy (Nx3)
        points_world = np.asarray(self.accumulated_points.points)
        ones = np.ones((points_world.shape[0], 1))
        points_world_hom = np.hstack([points_world, ones]).T  # 4xN

        # Transform world points to camera frame
        points_cam_hom = pose @ points_world_hom  # shape (4, N)
        points_cam = points_cam_hom[:3, :].T  # shape (N, 3)

        # Keep only points in front of camera
        z = points_cam[:, 2]
        valid = (z > self.min_range) & (z < self.max_range)
        points_cam = points_cam[valid]

        # Project to image plane
        x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        u = (x * fx / z + cx).astype(np.int32)
        v = (y * fy / z + cy).astype(np.int32)

        # Keep only valid pixel indices
        valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[valid_pixels]
        v = v[valid_pixels]
        valid_indices = np.where(valid)[0][valid_pixels]
        points_world_valid = points_world[valid_indices]

        # Sample RGB from the image
        rgb_values = rgb_image[v, u]  # shape (N, 3)
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
        points = list(
            zip(
                points_world_valid[:, 0],
                points_world_valid[:, 1],
                points_world_valid[:, 2],
                rgb_packed,
            )
        )
        pc2_msg = pc2.create_cloud(header, fields, points)
        self.point_cloud_pub.publish(pc2_msg)
        # Accumulate colored points
        colored_points = o3d.geometry.PointCloud()
        colored_points.points = o3d.utility.Vector3dVector(points_world_valid)
        colored_points.colors = o3d.utility.Vector3dVector(rgb_values / 255.0)
        self.colored_points += colored_points
        # Voxel downsample
        self.colored_points = self.colored_points.voxel_down_sample(
            voxel_size=self.voxel_size
        )

    def callback_signal(self, msg):
        save_path = Path(msg.data)
        if not save_path.parent.exists():
            rospy.logwarn(f"Path {save_path.parent} does not exist.")
            return
        if save_path.suffix != ".ply" and save_path.suffix != ".pcd":
            rospy.logwarn(f"Path {save_path} is not a .ply file.")
            return

        with self.lock:
            o3d.io.write_point_cloud(
                str(save_path),
                self.colored_points,
                write_ascii=True,
                compressed=False,
            )
        rospy.loginfo(f"Point cloud saved to {save_path}")


def main():
    rospy.init_node("d455_project_node", anonymous=True)
    d455_project = D455Project()
    rospy.spin()


if __name__ == "__main__":
    main()
