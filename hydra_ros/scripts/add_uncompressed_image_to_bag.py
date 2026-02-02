import rosbag
import cv_bridge
import argparse
from pathlib import Path
from sensor_msgs.msg import Image
import numpy as np
import cv2
import struct


def compressed_depth_to_cv2(msg: Image):
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
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
        depth_img_scaled = depthQuantA / (depth_img.astype(np.float32) - depthQuantB)
        # filter max values
        depth_img_scaled[depth_img == 0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img = (depth_img_scaled * 1000).astype(np.uint16)

    return depth_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", type=str, required=True)
    parser.add_argument("--compressed_topic", type=str, required=True)
    parser.add_argument("--uncompressed_topic", type=str, required=True)
    return parser.parse_args()


def main(bag_path: Path, compressed_topic: str, uncompressed_topic: str):
    assert bag_path.exists(), f"Bag path {bag_path} does not exist."
    bag = rosbag.Bag(bag_path)
    images = []

    bridge = cv_bridge.CvBridge()
    for topic, msg, t in bag.read_messages(topics=[compressed_topic]):
        if "depth" not in topic:
            image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
            uncompressed_msg = bridge.cv2_to_imgmsg(
                image, encoding="passthrough", header=msg.header
            )
        else:
            depth_img = compressed_depth_to_cv2(msg)
            uncompressed_msg = bridge.cv2_to_imgmsg(
                depth_img, encoding="passthrough", header=msg.header
            )
        images.append((uncompressed_msg, t))
    bag.close()

    bag = rosbag.Bag(bag_path, "a")
    for image, t in images:
        bag.write(uncompressed_topic, image, t)
    bag.close()


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.bag_path), args.compressed_topic, args.uncompressed_topic)
