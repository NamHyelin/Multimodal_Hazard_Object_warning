#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


DATE_TAG = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%m%d")


def read_xyz(msg: PointCloud2) -> np.ndarray:
    pts = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    pts = np.asarray(pts)

    if pts.dtype.fields is not None:
        xyz = np.column_stack(
            [
                np.asarray(pts["x"], dtype=np.float32),
                np.asarray(pts["y"], dtype=np.float32),
                np.asarray(pts["z"], dtype=np.float32),
            ]
        )
    else:
        xyz = np.asarray(pts, dtype=np.float32).reshape(-1, 3)

    return np.ascontiguousarray(xyz, dtype=np.float32)


class LidarBevSnapshot(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("lidar_bev_snapshot")
        self.args = args
        self.done = False

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.create_subscription(
            PointCloud2,
            args.topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Waiting for PointCloud2: {args.topic}")
        self.get_logger().info(
            f"BEV range x=[{args.x_min},{args.x_max}] "
            f"y=[{args.y_min},{args.y_max}] "
            f"z=[{args.z_min},{args.z_max}]"
        )

    def metric_to_pixel(self, x: float, y: float):
        a = self.args
        usable_w = a.width - 2 * a.margin
        usable_h = a.height - 2 * a.margin

        # Standard LiDAR BEV convention:
        # +x forward = image up
        # +y left    = image left
        # -y right   = image right
        u = a.margin + (a.y_max - y) / (a.y_max - a.y_min) * usable_w
        v = a.margin + (a.x_max - x) / (a.x_max - a.x_min) * usable_h
        return int(round(u)), int(round(v))

    def draw_grid(self, image: np.ndarray):
        a = self.args

        x_start = np.ceil(a.x_min / a.grid_m) * a.grid_m
        for x in np.arange(x_start, a.x_max + 1e-9, a.grid_m):
            u1, v = self.metric_to_pixel(float(x), a.y_min)
            u2, _ = self.metric_to_pixel(float(x), a.y_max)
            cv2.line(image, (u1, v), (u2, v), (210, 210, 210), 1)
            cv2.putText(
                image, f"x={x:.0f}", (8, max(18, v - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA
            )

        y_start = np.ceil(a.y_min / a.grid_m) * a.grid_m
        for y in np.arange(y_start, a.y_max + 1e-9, a.grid_m):
            u, v1 = self.metric_to_pixel(a.x_min, float(y))
            _, v2 = self.metric_to_pixel(a.x_max, float(y))
            cv2.line(image, (u, v1), (u, v2), (210, 210, 210), 1)
            cv2.putText(
                image, f"y={y:.0f}", (u + 3, a.height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (60, 60, 60), 1, cv2.LINE_AA
            )

        cv2.putText(
            image, "+x FORWARD", (a.width // 2 - 55, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            image, "+y LEFT", (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            image, "-y RIGHT", (a.width - 90, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA
        )

        if a.x_min <= 0.0 <= a.x_max and a.y_min <= 0.0 <= a.y_max:
            u0, v0 = self.metric_to_pixel(0.0, 0.0)
            cv2.circle(image, (u0, v0), 6, (0, 0, 0), 2)
            cv2.putText(
                image, "LiDAR origin (0,0)", (u0 + 8, v0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )

    def pointcloud_callback(self, msg: PointCloud2):
        if self.done:
            return

        try:
            xyz = read_xyz(msg)
        except Exception as exc:
            self.get_logger().error(f"Point cloud parsing failed: {exc}")
            return

        a = self.args
        valid = (
            np.isfinite(xyz).all(axis=1)
            & (xyz[:, 0] >= a.x_min)
            & (xyz[:, 0] <= a.x_max)
            & (xyz[:, 1] >= a.y_min)
            & (xyz[:, 1] <= a.y_max)
            & (xyz[:, 2] >= a.z_min)
            & (xyz[:, 2] <= a.z_max)
        )
        xyz = xyz[valid]

        if a.stride > 1:
            xyz = xyz[::a.stride]

        image = np.full((a.height, a.width, 3), 255, dtype=np.uint8)
        self.draw_grid(image)

        for x, y, z in xyz:
            u, v = self.metric_to_pixel(float(x), float(y))
            if 0 <= u < a.width and 0 <= v < a.height:
                radius = 1 if z < 0.8 else 2
                cv2.circle(image, (u, v), radius, (25, 25, 25), -1)

        stamp = msg.header.stamp
        stamp_s = f"{stamp.sec}.{stamp.nanosec:09d}"

        cv2.rectangle(image, (5, 30), (570, 78), (255, 255, 255), -1)
        cv2.putText(
            image,
            f"frame={msg.header.frame_id}  stamp={stamp_s}",
            (12, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            image,
            f"points shown={len(xyz)}  grid={a.grid_m:g} m",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA
        )

        out_path = self.output_dir / "bev_000000.png"
        npy_path = self.output_dir / "bev_000000_xyz.npy"

        cv2.imwrite(str(out_path), image)
        np.save(str(npy_path), xyz)

        print()
        print("=" * 80)
        print(f"[SAVE] {out_path}")
        print(f"[SAVE] {npy_path}")
        print(f"frame_id = {msg.header.frame_id}")
        print(f"points   = {len(xyz)}")
        print("BEV convention: +x=up/forward, +y=left, -y=right")
        print("=" * 80)

        self.done = True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display-free LiDAR BEV snapshot saver."
    )
    parser.add_argument(
        "--topic",
        default="/sensing/lidar/concatenated/pointcloud",
    )
    parser.add_argument(
        "--output_dir",
        default=f"/autoware_runtime/hyelin/results/lidar_bev_{DATE_TAG}",
    )

    # Defaults chosen around the user's current CenterPoint coordinates.
    parser.add_argument("--x_min", type=float, default=-5.0)
    parser.add_argument("--x_max", type=float, default=40.0)
    parser.add_argument("--y_min", type=float, default=-60.0)
    parser.add_argument("--y_max", type=float, default=20.0)

    parser.add_argument("--z_min", type=float, default=-2.0)
    parser.add_argument("--z_max", type=float, default=4.0)

    parser.add_argument("--grid_m", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--margin", type=int, default=60)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Draw every Nth filtered point.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = LidarBevSnapshot(args)

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
