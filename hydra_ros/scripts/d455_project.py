#!/usr/bin/env python3

import rospy
import cv_bridge
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
import tf2_ros

# import tf2_geometry_msgs
import numpy as np
import cv2
import struct

import tf.transformations


class D455Project:
    def __init__(self):
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw/compressedDepth", CompressedImage
        )
        self.color_sub = message_filters.Subscriber(
            "/camera/color/image_raw/compressed", CompressedImage
        )
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo
        )
        self.point_cloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=10)
        self.odom_pub = rospy.Publisher(
            "/msf_core/odometry_world", Odometry, queue_size=10
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cv_bridge = cv_bridge.CvBridge()

        self.sensor_frame = rospy.get_param("~sensor_frame", "body")
        # self.sensor_frame = rospy.get_param("~sensor_frame", "vn100_state_propogated")
        # self.world_frame = rospy.get_param("~world_frame", "camera_init")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.max_range = rospy.get_param("~max_range", 20.0)
        self.min_range = rospy.get_param("~min_range", 0.1)

        # Approximate time synchronizer
        # self.ts = message_filters.ApproximateTimeSynchronizer(
        #     [self.depth_sub, self.color_sub, self.camera_info_sub, self.odom_sub], 10, 0.1)
        self.ts = message_filters.TimeSynchronizer(
            [self.depth_sub, self.color_sub, self.camera_info_sub], 10
        )

        self.ts.registerCallback(self.callback_sync)
        rospy.loginfo("D455 Project node initialized.")

    @staticmethod
    def compressed_depth_to_cv2(msg: CompressedImage):
        # 'msg' as type CompressedImage
        depth_fmt, compr_type = msg.format.split(";")
        # remove white space
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()

        # remove header from raw data
        depth_header_size = 12
        raw_data = msg.data[depth_header_size:]

        depth_img = cv2.imdecode(
            np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
        )
        if depth_img is None:
            # probably wrong header size
            raise Exception(
                "Could not decode compressed depth image."
                "You may need to change 'depth_header_size'!"
            )

        if depth_fmt == "32FC1":
            raw_header = msg.data[:depth_header_size]
            # header: int, float, float
            [_, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
            depth_img_scaled = depthQuantA / (
                depth_img.astype(np.float32) - depthQuantB
            )
            # filter max values
            depth_img_scaled[depth_img == 0] = 0

            # depth_img_scaled provides distance in meters as f32
            # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
            depth_img = (depth_img_scaled * 1000).astype(np.uint16)

        return depth_img

    def callback_sync(
        self,
        depth_msg: CompressedImage,
        color_msg: CompressedImage,
        camera_info_msg: CameraInfo,
    ):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.sensor_frame,
                color_msg.header.stamp,
                rospy.Duration(1.0),  # wait up to 1 second
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logwarn("Failed to get sensor pose in world frame.")
            print
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
        depth_image = self.compressed_depth_to_cv2(depth_msg)
        # print(cv2.imwrite("/home/arl/Desktop/depth_image.png", depth_image))
        depth_image = np.array(depth_image, dtype=np.float32) / 1000.0
        # Get the color image.
        color_image = self.cv_bridge.compressed_imgmsg_to_cv2(color_msg, "rgb8")
        color_image = np.array(color_image, dtype=np.uint8)
        # Get the camera intrinsics.
        intrinsics = np.array(camera_info_msg.K).reshape(3, 3)

        self.publish_pointcloud(
            camera_info_msg.header, depth_image, color_image, pose, intrinsics
        )

        # Publish the odometry message to the world frame.
        odom_msg = Odometry()
        odom_msg.header = camera_info_msg.header
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.sensor_frame
        odom_msg.pose.pose.position.x = pose[0, 3]
        odom_msg.pose.pose.position.y = pose[1, 3]
        odom_msg.pose.pose.position.z = pose[2, 3]
        odom_msg.pose.pose.orientation.x = transform.transform.rotation.x
        odom_msg.pose.pose.orientation.y = transform.transform.rotation.y
        odom_msg.pose.pose.orientation.z = transform.transform.rotation.z
        odom_msg.pose.pose.orientation.w = transform.transform.rotation.w
        self.odom_pub.publish(odom_msg)

    def callback(
        self,
        depth_msg: CompressedImage,
        color_msg: CompressedImage,
        camera_info_msg: CameraInfo,
        odom_msg: Odometry,
    ):
        # Get pose of the sensor in the world frame using tf.
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.sensor_frame, rospy.Time(0)
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

        # C = np.eye(4)
        # C[1, 1] = -1
        # C[2, 2] = -1
        # pose = np.matmul(pose, C)

        # # Get the pose of the sensor in the world frame using odometry.
        # pose = np.eye(4)
        # pose[:3, 3] = [odom_msg.pose.pose.position.x,
        #                odom_msg.pose.pose.position.y,
        #                odom_msg.pose.pose.position.z]
        # q = [odom_msg.pose.pose.orientation.x,
        #      odom_msg.pose.pose.orientation.y,
        #      odom_msg.pose.pose.orientation.z,
        #      odom_msg.pose.pose.orientation.w]
        # # Convert quaternion to rotation matrix.
        # q = [q[0], q[1], q[2], q[3]]
        # R = tf.transformations.quaternion_matrix(q)[:3, :3]
        # pose[:3, :3] = R

        # Get the depth image.
        depth_image = self.compressed_depth_to_cv2(depth_msg)
        depth_image = np.array(depth_image, dtype=np.float32) / 1000.0
        # Get the color image.
        color_image = self.cv_bridge.compressed_imgmsg_to_cv2(color_msg, "rgb8")
        color_image = np.array(color_image, dtype=np.uint8)
        # Get the camera intrinsics.
        intrinsics = np.array(camera_info_msg.K).reshape(3, 3)

        self.publish_pointcloud(
            camera_info_msg.header, depth_image, color_image, pose, intrinsics
        )

        # Publish the odometry message to the world frame.
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.sensor_frame
        self.odom_pub.publish(odom_msg)

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
        mask = depth > 0
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
        header.frame_id = self.world_frame
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
