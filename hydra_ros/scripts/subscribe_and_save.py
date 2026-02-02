#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class CompressedImageSaver:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("compressed_image_saver", anonymous=True)

        # Create CvBridge instance
        self.bridge = CvBridge()

        # Output directory for saved images
        self.output_dir = rospy.get_param("~output_dir", "/home/arl/jetson_ssd/images")
        os.makedirs(self.output_dir, exist_ok=True)

        # Topic to subscribe to
        image_topic = rospy.get_param(
            "~image_topic", "/camera/color/image_raw/compressed"
        )

        # Subscribe to the compressed image topic
        rospy.Subscriber(image_topic, CompressedImage, self.image_callback)

        rospy.loginfo(f"Subscribed to {image_topic}")
        rospy.loginfo(f"Saving images to {self.output_dir}")

    def image_callback(self, msg):
        try:
            # Convert the compressed ROS image to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.output_dir, f"image_{timestamp}.jpg")

            # Save image
            cv2.imwrite(filename, cv_image)
            rospy.loginfo(f"Saved image: {filename}")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


if __name__ == "__main__":
    try:
        CompressedImageSaver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
