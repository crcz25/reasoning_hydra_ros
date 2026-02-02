import rosbag
import rospy
from sensor_msgs.msg import Image
import bisect
import cv_bridge
import argparse
from pathlib import Path
import numpy as np
import cv2
import struct
from nav_msgs.msg import Odometry


def main(
    bag_path: Path,
    output_bag: Path,
    depth_compressed_topic: str,
    rgb_compressed_topic: str,
    camera_info_topic: str,
    depth_info_topic: str,
    depth_uncompressed_topic: str,
    rgb_uncompressed_topic: str,
    odometry_topic: str,
    time_tolerance: float,
):
    assert bag_path.exists(), f"Bag path {bag_path} does not exist."
    bridge = cv_bridge.CvBridge()
    depth_images_compressed = []
    rgb_images_compressed = []
    camera_infos = []
    depth_infos = []
    odometrys = []
    tfs = []
    static_tfs = []
    time_tolerance = rospy.Duration(time_tolerance)
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(
            topics=[
                depth_compressed_topic,
                rgb_compressed_topic,
                camera_info_topic,
                depth_info_topic,
                odometry_topic,
                "/tf",
                "/tf_static",
            ]
        ):
            if topic == depth_compressed_topic:
                depth_images_compressed.append((msg.header.stamp, msg))
            elif topic == rgb_compressed_topic:
                rgb_images_compressed.append((msg.header.stamp, msg))
            elif topic == camera_info_topic:
                camera_infos.append((msg.header.stamp, msg))
            elif topic == depth_info_topic:
                depth_infos.append((msg.header.stamp, msg))
            elif topic == odometry_topic:
                odometrys.append((msg.header.stamp, msg))
            elif topic == "/tf":
                tfs.append((t, msg))
            elif topic == "/tf_static":
                static_tfs.append((t, msg))
    # Sort messages by timestamp
    depth_images_compressed.sort(key=lambda x: x[0])
    rgb_images_compressed.sort(key=lambda x: x[0])
    camera_infos.sort(key=lambda x: x[0])
    depth_infos.sort(key=lambda x: x[0])
    odometrys.sort(key=lambda x: x[0])

    # Find closest matches
    synced_msgs = []
    depth_times = [t for t, _ in depth_images_compressed]  # Extract just the timestamps
    odometry_times = [t for t, _ in odometrys]  # Extract just the timestamps

    i = 0
    for rgb_time, rgb_msg in rgb_images_compressed:
        idx = bisect.bisect_left(depth_times, rgb_time)  # Find closest match
        idx_odom = bisect.bisect_left(odometry_times, rgb_time)  # Find closest match

        # Check neighbors for the best match
        best_match = None
        best_match_info = None
        best_match_odom = None
        if idx < len(depth_times):
            best_match = depth_images_compressed[idx]  # Closest future match
            best_match_info = depth_infos[idx][1]
        if idx > 0 and (
            best_match is None
            or abs(depth_times[idx - 1] - rgb_time) < abs(best_match[0] - rgb_time)
        ):
            best_match = depth_images_compressed[idx - 1]  # Closest past match
            best_match_info = depth_infos[idx - 1][1]
        if idx_odom < len(odometry_times):
            best_match_odom = odometrys[idx_odom]
        if idx_odom > 0 and (
            best_match_odom is None
            or abs(odometry_times[idx_odom - 1] - rgb_time)
            < abs(best_match_odom[0] - rgb_time)
        ):
            best_match_odom = odometrys[idx_odom - 1]

        # If a valid match is found within tolerance, store it
        if (best_match and abs(best_match[0] - rgb_time) <= time_tolerance) and (
            best_match_odom and abs(best_match_odom[0] - rgb_time) <= time_tolerance
        ):
            synced_msgs.append(
                (
                    rgb_time,
                    rgb_msg,
                    best_match[1],
                    camera_infos[i][1],
                    best_match_info,
                    best_match_odom[1],
                )
            )
        i += 1

    # Save the synchronized messages into a new bag file
    with rosbag.Bag(output_bag, "w") as out_bag:
        for t, rgb_msg, depth_msg, camera_info, depth_info, odom in synced_msgs:
            image_uncompressed = bridge.compressed_imgmsg_to_cv2(
                rgb_msg, desired_encoding="passthrough"
            )
            image_msg_uncompressed = bridge.cv2_to_imgmsg(
                image_uncompressed, encoding="passthrough", header=rgb_msg.header
            )

            depth_uncompressed = compressed_depth_to_cv2(depth_msg)
            depth_msg_uncompressed = bridge.cv2_to_imgmsg(
                depth_uncompressed, encoding="passthrough", header=rgb_msg.header
            )

            odom.header.stamp = camera_info.header.stamp
            odom.header.frame_id = "mimosa_world"
            odom.child_frame_id = "mimosa_body"

            out_bag.write(rgb_uncompressed_topic, image_msg_uncompressed, t)
            out_bag.write(depth_uncompressed_topic, depth_msg_uncompressed, t)
            out_bag.write(camera_info_topic, camera_info, t)
            depth_info.header.stamp = camera_info.header.stamp
            out_bag.write(depth_info_topic, depth_info, t)
            out_bag.write(odometry_topic, odom, t)

        # Write static transforms
        for t, msg in static_tfs:
            out_bag.write("/tf_static", msg, t)
        # Write dynamic transforms
        for t, msg in tfs:
            out_bag.write("/tf", msg, t)
    # Print the number of synchronized messages
    print(f"Saved synchronized bag with {len(synced_msgs)} pairs.")


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
    parser.add_argument("--output_bag", type=str, required=True)
    parser.add_argument("--depth_compressed", type=str, required=True)
    parser.add_argument("--rgb_compressed", type=str, required=True)
    parser.add_argument("--camera_info", type=str, required=True)
    parser.add_argument("--depth_info", type=str, required=True)
    parser.add_argument("--depth_uncompressed", type=str, required=True)
    parser.add_argument("--rgb_uncompressed", type=str, required=True)
    parser.add_argument("--time_tolerance", type=float, default=0.01)
    parser.add_argument("--odometry_topic", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        Path(args.bag_path),
        Path(args.output_bag),
        args.depth_compressed,
        args.rgb_compressed,
        args.camera_info,
        args.depth_info,
        args.depth_uncompressed,
        args.rgb_uncompressed,
        args.odometry_topic,
        args.time_tolerance,
    )
