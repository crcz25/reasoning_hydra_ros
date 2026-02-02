#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import os


class CompressedVideoSaverTimeConsistent:
    def __init__(self):
        self.output_dir = rospy.get_param(
            "~output_dir", "/home/arl/jetson_ssd/videos/clear_exits/2"
        )
        self.topic = rospy.get_param("~topic", "/camera/color/image_raw/compressed")
        self.codec = rospy.get_param("~codec", "mp4v")
        self.output_path = os.path.join(self.output_dir, f"d455.mp4")

        self.writer = None
        self.frame_size = None
        self.last_stamp = None
        self.last_frame = None
        self.estimated_fps = None
        self.frame_count = 0

        rospy.loginfo(f"Saving time-consistent compressed video to: {self.output_path}")
        rospy.Subscriber(self.topic, CompressedImage, self.callback, queue_size=10)

    def callback(self, msg):
        # Decode JPEG/PNG from ROS CompressedImage
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn("Failed to decode compressed image frame.")
            return

        # First frame: just store timestamp
        if self.last_stamp is None:
            self.last_stamp = msg.header.stamp
            self.last_frame = frame
            self.frame_count = 1
            return

        # If FPS not yet estimated, calculate it from the first two frames
        if self.estimated_fps is None:
            dt = (msg.header.stamp - self.last_stamp).to_sec()
            if dt <= 0:
                rospy.logwarn("Non-positive time difference, skipping frame.")
                return
            self.estimated_fps = 1.0 / dt
            self.frame_size = (frame.shape[1], frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.estimated_fps, self.frame_size
            )
            if not self.writer.isOpened():
                rospy.logerr("Failed to open video writer.")
                rospy.signal_shutdown("Video writer error")
                return

            # Write first stored frame and current frame
            self.writer.write(self.last_frame)
            self.writer.write(frame)
            self.last_stamp = msg.header.stamp
            self.last_frame = frame
            return

        # After FPS is set, fill in any gap frames
        dt = (msg.header.stamp - self.last_stamp).to_sec()
        frame_gap = int(round(dt * self.estimated_fps)) - 1
        if frame_gap > 0 and self.last_frame is not None:
            for _ in range(frame_gap):
                self.writer.write(self.last_frame)

        # Write current frame
        self.writer.write(frame)

        # Update state
        self.last_stamp = msg.header.stamp
        self.last_frame = frame

    def shutdown(self):
        rospy.loginfo("Shutting down and releasing video writer.")
        if self.writer:
            self.writer.release()


if __name__ == "__main__":
    rospy.init_node("compressed_video_saver_time_consistent", anonymous=True)
    saver = CompressedVideoSaverTimeConsistent()
    rospy.on_shutdown(saver.shutdown)
    rospy.spin()
