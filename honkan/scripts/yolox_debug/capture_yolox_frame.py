#!/usr/bin/env python3

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


TOPICS = {
    "gate": "/tensorrt_yolox_gate/out/image",
    "intersection": "/tensorrt_yolox_intersection/out/image",
}


class YoloXFrameCapture(Node):
    def __init__(self) -> None:
        super().__init__("yolox_frame_capture")
        self.frames = {}
        self._image_subscriptions = [
            self.create_subscription(
                Image,
                topic,
                lambda message, camera=camera: self.frames.setdefault(camera, message),
                qos_profile_sensor_data,
            )
            for camera, topic in TOPICS.items()
        ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one annotated YOLOX image from each Taki camera."
    )
    parser.add_argument(
        "--output-dir",
        default="/autoware_runtime/captures/yolox",
        help="Directory in which the two PNG files are written.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for both image topics (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    rclpy.init()
    node = YoloXFrameCapture()
    deadline = time.monotonic() + args.timeout

    try:
        while rclpy.ok() and len(node.frames) < len(TOPICS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = [TOPICS[name] for name in TOPICS if name not in node.frames]
                node.get_logger().error("Timed out waiting for: " + ", ".join(missing))
                return 1
            rclpy.spin_once(node, timeout_sec=min(0.2, remaining))

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        bridge = CvBridge()

        for camera, message in node.frames.items():
            image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            output_path = output_dir / f"{camera}_yolox.png"
            if not cv2.imwrite(str(output_path), image):
                raise RuntimeError(f"Failed to write {output_path}")
            print(output_path)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
