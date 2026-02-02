#!/usr/bin/env python3
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
import tf
import cv_bridge
from std_msgs.msg import String
from PIL import Image as PILImage
import cv2

import tf.transformations


class RGBPCL:
    def __init__(self):
        # Synchronize detph and color. Pose is queried with tf.
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw/uncompressed", Image
        )
        self.color_sub = message_filters.Subscriber(
            "/camera/color/image_raw/image_raw", Image
        )
        self.odometry_sub = message_filters.Subscriber("/msf_core/odometry", Odometry)
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo
        )

        self.ts = message_filters.TimeSynchronizer(
            [self.depth_sub, self.color_sub, self.camera_info_sub, self.odometry_sub],
            10,
        )
        self.ts.registerCallback(self.callback)
        self.pointcloud_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=1)
        self.tf_listener = tf.TransformListener()

        self.cv_bridge = cv_bridge.CvBridge()

        self.sensor_frame = rospy.get_param("~sensor_frame", "mimosa_body")
        self.world_frame = rospy.get_param("~world_frame", "mimosa_world")
        rospy.loginfo("RGBPCL node initialized.")

    def callback(
        self,
        depth_msg: Image,
        color_msg: Image,
        camera_info_msg: CameraInfo,
        odom: Odometry,
    ):
        # Get pose of the sensor in the world frame using tf.
        # try:
        #     (trans, rot) = self.tf_listener.lookupTransform(
        #         self.world_frame, self.sensor_frame, rospy.Time(0)
        #     )
        # except (
        #     tf.LookupException,
        #     tf.ConnectivityException,
        #     tf.ExtrapolationException,
        # ):
        #     rospy.logwarn("Failed to get sensor pose in world frame.")
        #     return

        # rospy.loginfo("Received depth and color images.")
        # # Pose in homogeneous transformation matrix.
        # pose = np.eye(4)
        # pose[:3, 3] = trans
        # pose[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]

        # Get the pose from the odometry message.
        pose = np.eye(4)
        pose[:3, 3] = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ]
        pose[:3, :3] = tf.transformations.quaternion_matrix(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ]
        )[:3, :3]

        # Get the depth image.
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        depth_image = (
            np.array(depth_image, dtype=np.float32) / 1000.0
        )  # Convert to meters

        # Get the color image.
        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = np.array(color_image, dtype=np.uint8)
        PILImage.fromarray(color_image).save("/home/albert/Desktop/color.png")

        # Get the camera intrinsics.
        intrinsics = np.array(camera_info_msg.K).reshape(3, 3)
        # Convert the depth image to a point cloud and publish it.
        self.publish_pointcloud(
            depth_msg.header, depth_image, color_image, pose, intrinsics
        )

    def publish_pointcloud(self, header, depth_image, rgb_image, pose, intrinsics):
        header.frame_id = self.world_frame
        height, width = depth_image.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

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
    rospy.init_node("rgb_pcl_pub", anonymous=True)
    publisher = RGBPCL()
    rospy.spin()


if __name__ == "__main__":
    main()
