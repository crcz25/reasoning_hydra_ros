#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os


class RawVideoSaverTimeConsistent:
    def __init__(self):
        self.output_dir = rospy.get_param(
            "~output_dir", "/home/arl/jetson_ssd/videos/clear_exits/2"
        )
        self.topic = rospy.get_param("~topic", "/semantic_inference/detections")
        self.codec = rospy.get_param("~codec", "mp4v")  # mp4 output
        self.fill_empty_with_black = rospy.get_param("~fill_empty_with_black", False)

        self.output_path = os.path.join(self.output_dir, "detections.mp4")

        self.bridge = CvBridge()
        self.writer = None
        self.frame_size = None
        self.last_stamp = None
        self.last_frame = None
        self.estimated_fps = None

        rospy.loginfo(f"Saving time-consistent raw video to: {self.output_path}")
        rospy.Subscriber(self.topic, Image, self.callback, queue_size=10)

    def callback(self, msg):
        print(f"Received frame at time: {msg.header.stamp.to_sec()}")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"Failed to convert Image message: {e}")
            return

        # First frame: store timestamp & frame
        if self.last_stamp is None:
            self.last_stamp = msg.header.stamp
            self.last_frame = frame.copy()
            return

        # If FPS not yet estimated, calculate from first two frames
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
            self.last_frame = frame.copy()
            return

        # After FPS is known: fill gap if needed
        dt = (msg.header.stamp - self.last_stamp).to_sec()
        frame_gap = int(round(dt * self.estimated_fps)) - 1
        if frame_gap > 0:
            filler_frame = (
                self.last_frame
                if not self.fill_empty_with_black
                else (0 * self.last_frame)
            )  # black frame
            for _ in range(frame_gap):
                self.writer.write(filler_frame)

        # Write current frame
        self.writer.write(frame)

        # Update state
        self.last_stamp = msg.header.stamp
        self.last_frame = frame.copy()

    def shutdown(self):
        rospy.loginfo("Shutting down and releasing video writer.")
        if self.writer:
            self.writer.release()


if __name__ == "__main__":
    rospy.init_node("raw_video_saver_time_consistent", anonymous=True)
    saver = RawVideoSaverTimeConsistent()
    rospy.on_shutdown(saver.shutdown)
    rospy.spin()
