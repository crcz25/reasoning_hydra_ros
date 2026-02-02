#!/usr/bin/env python3

import rospy
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import numpy as np
import cv2

import tf.transformations


class D455Project:
    def __init__(self):
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw/image_raw", Image
        )
        self.color_sub = message_filters.Subscriber(
            "/camera/color/image_raw/image_raw", Image
        )
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo
        )
        self.point_cloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = cv_bridge.CvBridge()

        self.sensor_frame = rospy.get_param("~sensor_frame", "mimosa_body_cam")
        # self.sensor_frame = rospy.get_param("~sensor_frame", "vn100_state_propogated")
        # self.world_frame = rospy.get_param("~world_frame", "camera_init")
        self.world_frame = rospy.get_param("~world_frame", "mimosa_world")
        self.min_range = rospy.get_param("~min_range", 0.1)
        self.max_range = rospy.get_param("~max_range", 5.0)

        # Approximate time synchronizer
        # self.ts = message_filters.ApproximateTimeSynchronizer(
        #     [self.depth_sub, self.color_sub, self.camera_info_sub, self.odom_sub], 10, 0.1)
        self.ts = message_filters.TimeSynchronizer(
            [self.depth_sub, self.color_sub, self.camera_info_sub], 10
        )

        self.ts.registerCallback(self.callback_sync)
        rospy.loginfo("D455 Project node initialized.")

    def callback_sync(
        self, depth_msg: Image, color_msg: Image, camera_info_msg: CameraInfo
    ):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.sensor_frame, color_msg.header.stamp
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Failed to get sensor pose in world frame.")
            return

        rospy.loginfo("Received depth and color images.")
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
        R = tf.transformations.quaternion_matrix(q)[:3, :3]
        pose[:3, :3] = R
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        depth_image = np.array(depth_image, dtype=np.float32)
        # Get the color image.
        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, "passthrough")
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = np.array(color_image, dtype=np.uint8)
        # Get the camera intrinsics.
        intrinsics = np.array(camera_info_msg.K).reshape(3, 3)

        self.publish_pointcloud_direct(
            camera_info_msg.header, depth_image, color_image, pose, intrinsics
        )

    def publish_pointcloud_direct(
        self, header, depth_image, rgb_image, pose, instrinsics
    ):
        header.frame_id = "mimosa_world"
        # header.frame_id = self.world_frame
        height, width = depth_image.shape
        fx, fy = instrinsics[0, 0], instrinsics[1, 1]
        cx, cy = instrinsics[0, 2], instrinsics[1, 2]

        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u, v = u.flatten(), v.flatten()
        depth = depth_image.flatten()

        # Filter out points with zero depth
        mask = (depth > self.min_range) & (depth < self.max_range)
        u, v, depth = u[mask], v[mask], depth[mask]

        # Convert pixel coordinates to camera frame (3D points in optical frame)
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points_camera_optical = np.vstack(
            (x, y, z, np.ones_like(x))
        )  # Homogeneous coordinates

        # Transform  points to world coordinates
        points_world = pose @ points_camera_optical
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
        self.point_cloud_pub.publish(pc2_msg)

    def publish_pointcloud(self, header, depth_image, rgb_image, pose, instrinsics):
        header.frame_id = "mimosa_world"
        # header.frame_id = self.world_frame
        height, width = depth_image.shape
        fx, fy = instrinsics[0, 0], instrinsics[1, 1]
        cx, cy = instrinsics[0, 2], instrinsics[1, 2]

        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u, v = u.flatten(), v.flatten()
        depth = depth_image.flatten()

        # Filter out points with zero depth
        mask = depth > 0
        u, v, depth = u[mask], v[mask], depth[mask]

        # Convert pixel coordinates to camera frame (3D points in optical frame)
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points_camera_optical = np.vstack(
            (x, y, z, np.ones_like(x))
        )  # Homogeneous coordinates

        # FLU transformation (convert from D455 optical frame to FLU)
        R_optical_to_flu = np.array(
            [
                [0, 0, 1],  # X_FLU = Z_optical
                [-1, 0, 0],  # Y_FLU = -X_optical
                [0, -1, 0],
            ]
        )  # Z_FLU = -Y_optical
        points_flu = np.vstack(
            (R_optical_to_flu @ points_camera_optical[:3, :], np.ones_like(x))
        )

        # Transform FLU points to world coordinates
        points_world = pose @ points_flu
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
        self.point_cloud_pub.publish(pc2_msg)


def main():
    rospy.init_node("d455_project_node", anonymous=True)
    d455_project = D455Project()
    rospy.spin()


if __name__ == "__main__":
    main()
