#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
import tf
import cv_bridge
from std_msgs.msg import String
from pathlib import Path

import tf.transformations


class TSDF:
    def __init__(self):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        # Synchronize detph and color. Pose is queried with tf.
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw/image_raw", Image
        )
        self.color_sub = message_filters.Subscriber(
            "/camera/color/image_raw/image_raw", Image
        )
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo
        )

        self.ts = message_filters.TimeSynchronizer(
            [self.depth_sub, self.color_sub, self.camera_info_sub], 10
        )
        self.ts.registerCallback(self.callback)
        self.signal_sub = rospy.Subscriber("/tsdf_signal", String, self.signal_callback)
        self.tf_listener = tf.TransformListener()

        self.cv_bridge = cv_bridge.CvBridge()

        self.sensor_frame = rospy.get_param("~sensor_frame", "mimosa_body_cam")
        self.world_frame = rospy.get_param("~world_frame", "mimosa_world")
        rospy.loginfo("TSDF node initialized.")

    def callback(
        self,
        depth_msg: Image,
        color_msg: Image,
        camera_info_msg: CameraInfo,
    ):
        # Get pose of the sensor in the world frame using tf.
        try:
            (trans, rot) = self.tf_listener.lookupTransform(
                self.world_frame, self.sensor_frame, camera_info_msg.header.stamp
            )
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            rospy.logwarn("Failed to get sensor pose in world frame.")
            return

        rospy.loginfo("Received depth and color images.")
        # Pose in homogeneous transformation matrix.
        pose = np.eye(4)
        pose[:3, 3] = trans
        pose[:3, :3] = tf.transformations.quaternion_matrix(rot)[:3, :3]

        # Get the depth image.
        depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg)
        depth_image = np.array(depth_image, dtype=np.float32)

        # Get the color image.
        color_image = self.cv_bridge.imgmsg_to_cv2(color_msg)
        color_image = np.array(color_image, dtype=np.uint8)

        # Get the camera intrinsics.
        intrinshics = o3d.camera.PinholeCameraIntrinsic(
            depth_image.shape[1],
            depth_image.shape[0],
            camera_info_msg.K[0],
            camera_info_msg.K[4],
            camera_info_msg.K[2],
            camera_info_msg.K[5],
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image((depth_image * 1000).astype(np.uint16)),
            convert_rgb_to_intensity=False,
            depth_trunc=10.0,
        )

        self.volume.integrate(rgbd_image, intrinshics, np.linalg.inv(pose))

    def signal_callback(self, msg: String):
        rospy.loginfo("Received signal.")
        path = Path(msg.data)
        if path.parent.exists():
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(str(path), mesh)
            rospy.loginfo(f"Mesh saved to {path}.")
        else:
            rospy.logwarn("Path does not exist.")


def main():
    rospy.init_node("tsdf")
    tsdf = TSDF()
    rospy.spin()


if __name__ == "__main__":
    main()
