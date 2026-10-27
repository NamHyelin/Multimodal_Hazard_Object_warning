#!/usr/bin/env python3

"""Capture raw gate-camera frames and synchronized Autoware YOLOX detections."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tier4_perception_msgs.msg import DetectedObjectsWithFeature


LABELS = {
    0: "UNKNOWN",
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "TRAILER",
    5: "MOTORCYCLE",
    6: "BICYCLE",
    7: "PEDESTRIAN",
}


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class BenchmarkCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("yolo_benchmark_capture")
        self.args = args
        self.bridge = CvBridge()
        self.images = deque(maxlen=120)
        self.records = []
        self.last_saved_stamp_ns = None
        self.done = False

        self.output_dir = Path(args.output_dir)
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(
            Image, args.image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            DetectedObjectsWithFeature,
            args.yolox_topic,
            self.yolox_callback,
            qos_profile_sensor_data,
        )

    def image_callback(self, msg: Image) -> None:
        self.images.append((stamp_ns(msg.header.stamp), msg))

    def yolox_callback(self, msg: DetectedObjectsWithFeature) -> None:
        target_ns = stamp_ns(msg.header.stamp)
        min_period_ns = int(1_000_000_000 / self.args.sample_hz)
        if (
            self.last_saved_stamp_ns is not None
            and target_ns - self.last_saved_stamp_ns < min_period_ns
        ):
            return
        if not self.images:
            return

        image_stamp_ns, image_msg = min(
            self.images, key=lambda item: abs(item[0] - target_ns)
        )
        delta_ms = abs(image_stamp_ns - target_ns) / 1_000_000.0
        if delta_ms > self.args.max_sync_ms:
            return

        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        index = len(self.records)
        filename = f"frame_{index:04d}_{target_ns}.jpg"
        path = self.image_dir / filename
        if not cv2.imwrite(
            str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
        ):
            raise RuntimeError(f"Failed to write {path}")

        detections = []
        for feature_object in msg.feature_objects:
            classifications = feature_object.object.classification
            if classifications:
                best = max(classifications, key=lambda item: float(item.probability))
                label = int(best.label)
                class_probability = float(best.probability)
            else:
                label = 0
                class_probability = None

            roi = feature_object.feature.roi
            detections.append(
                {
                    "label_id": label,
                    "class_name": LABELS.get(label, f"LABEL_{label}"),
                    "confidence": float(feature_object.object.existence_probability),
                    "class_probability": class_probability,
                    "bbox_xyxy": [
                        int(roi.x_offset),
                        int(roi.y_offset),
                        int(roi.x_offset + roi.width),
                        int(roi.y_offset + roi.height),
                    ],
                }
            )

        receive_wall_ns = time.time_ns()
        self.records.append(
            {
                "index": index,
                "image": f"images/{filename}",
                "stamp_ns": target_ns,
                "frame_id": msg.header.frame_id,
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "sync_delta_ms": delta_ms,
                "yolox_output_age_ms": (receive_wall_ns - target_ns) / 1_000_000.0,
                "yolox_detections": detections,
            }
        )
        self.last_saved_stamp_ns = target_ns
        print(
            f"Captured {len(self.records)}/{self.args.frames}: {filename}, "
            f"YOLOX objects={len(detections)}, age={self.records[-1]['yolox_output_age_ms']:.1f} ms",
            flush=True,
        )
        if len(self.records) >= self.args.frames:
            self.done = True

    def write_manifest(self) -> None:
        manifest = {
            "image_topic": self.args.image_topic,
            "yolox_topic": self.args.yolox_topic,
            "sample_hz": self.args.sample_hz,
            "frames": self.records,
        }
        (self.output_dir / "capture.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--max-sync-ms", type=float, default=100.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--image-topic", default="/rsu1/gate/image_raw")
    parser.add_argument("--yolox-topic", default="/rsu1/gate/yolox/objects")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0 or args.sample_hz <= 0 or args.timeout <= 0:
        raise ValueError("frames, sample-hz, and timeout must be positive")

    rclpy.init()
    node = BenchmarkCapture(args)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.write_manifest()
        if not node.done:
            node.get_logger().error(
                f"Timed out after capturing {len(node.records)}/{args.frames} frames"
            )
            return 1
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
