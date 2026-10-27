#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

import rclpy
from autoware_perception_msgs.msg import DetectedObjects
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs.msg import Image as RosImage
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from tier4_perception_msgs.msg import DetectedObjectsWithFeature
except Exception:
    DetectedObjectsWithFeature = None


DATE_TAG = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%m%d")


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ros_image_to_bgr(msg: RosImage) -> np.ndarray:
    enc = msg.encoding.lower()
    if enc not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()
    if enc == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def quaternion_to_rotation_matrix(x, y, z, w):
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def transform_to_matrix(transform):
    q = transform.rotation
    t = transform.translation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def read_xyz(msg: PointCloud2) -> np.ndarray:
    pts = point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True
    )
    pts = np.asarray(pts)
    if pts.dtype.fields is not None:
        xyz = np.column_stack([
            np.asarray(pts["x"], dtype=np.float32),
            np.asarray(pts["y"], dtype=np.float32),
            np.asarray(pts["z"], dtype=np.float32),
        ])
    else:
        xyz = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    return np.ascontiguousarray(xyz, dtype=np.float64)


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float64)
    ph = np.column_stack([xyz, np.ones(len(xyz), dtype=np.float64)])
    return (T @ ph.T).T[:, :3]


class ProjectionIsolationDebug(Node):
    """
    Isolates where the mismatch comes from.

    Output overlay:
      CYAN dots   : raw /rsu1 LiDAR projected directly to the selected camera
      YELLOW dots : concatenated lidar_fusion cloud projected to the camera
      RED crosses : CenterPoint 3D centers projected to the camera
      GREEN boxes : YOLOX 2D detections (camera-side reference)
    """

    def __init__(self, args):
        super().__init__("projection_isolation_debug")

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.latest_raw = None
        self.latest_fused = None
        self.latest_centerpoint = None
        self.latest_yolox = None
        self.last_save_wall = 0.0
        self.frame_index = 0

        self.K = None
        self.D = None
        self.camera_frame = None
        self.calib_width = None
        self.calib_height = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        cb = ReentrantCallbackGroup()

        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_cb,
            qos_profile_sensor_data,
            callback_group=cb,
        )
        self.create_subscription(
            PointCloud2,
            args.raw_pointcloud_topic,
            self.raw_cloud_cb,
            qos_profile_sensor_data,
            callback_group=cb,
        )
        self.create_subscription(
            PointCloud2,
            args.fused_pointcloud_topic,
            self.fused_cloud_cb,
            qos_profile_sensor_data,
            callback_group=cb,
        )
        self.create_subscription(
            DetectedObjects,
            args.centerpoint_topic,
            self.centerpoint_cb,
            qos_profile_sensor_data,
            callback_group=cb,
        )

        if DetectedObjectsWithFeature is not None:
            self.create_subscription(
                DetectedObjectsWithFeature,
                args.yolox_topic,
                self.yolox_cb,
                qos_profile_sensor_data,
                callback_group=cb,
            )

        self.create_subscription(
            RosImage,
            args.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
            callback_group=cb,
        )

        self.get_logger().info("Projection isolation debug started.")
        self.get_logger().info(f"Output: {self.output_dir}")

    def camera_info_cb(self, msg):
        with self.lock:
            self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.asarray(msg.d, dtype=np.float64).reshape(-1)
            self.camera_frame = msg.header.frame_id
            self.calib_width = int(msg.width)
            self.calib_height = int(msg.height)

    def _pack(self, msg):
        return {
            "msg": msg,
            "frame_id": msg.header.frame_id,
            "stamp_ns": stamp_ns(msg.header.stamp),
            "arrival_ns": time.time_ns(),
        }

    def raw_cloud_cb(self, msg):
        with self.lock:
            self.latest_raw = self._pack(msg)

    def fused_cloud_cb(self, msg):
        with self.lock:
            self.latest_fused = self._pack(msg)

    def centerpoint_cb(self, msg):
        with self.lock:
            self.latest_centerpoint = self._pack(msg)

    def yolox_cb(self, msg):
        with self.lock:
            self.latest_yolox = self._pack(msg)

    def lookup_static(self, target_frame, source_frame):
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.5),
            )
            return transform_to_matrix(tf.transform)
        except TransformException as exc:
            self.get_logger().warning(
                f"TF failed {source_frame}->{target_frame}: {exc}"
            )
            return None

    def project_cloud(self, packed, K, D, camera_frame, image_shape, stride=30):
        if packed is None:
            return np.empty((0, 2), dtype=np.float64), "missing"

        msg = packed["msg"]
        try:
            xyz = read_xyz(msg)
        except Exception as exc:
            return np.empty((0, 2), dtype=np.float64), f"parse:{exc}"

        xyz = xyz[::max(1, stride)]
        T = self.lookup_static(camera_frame, packed["frame_id"])
        if T is None:
            return np.empty((0, 2), dtype=np.float64), "tf"

        cam = transform_points(xyz, T)
        valid = (
            np.isfinite(cam).all(axis=1)
            & (cam[:, 2] > 0.1)
            & (cam[:, 2] < 120.0)
        )
        cam = cam[valid]
        if len(cam) == 0:
            return np.empty((0, 2), dtype=np.float64), "no-front-points"

        uv, _ = cv2.projectPoints(
            cam.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            D,
        )
        uv = uv.reshape(-1, 2)

        h, w = image_shape[:2]
        inside = (
            np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0) & (uv[:, 0] < w)
            & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        return uv[inside], "ok"

    def draw_centerpoint_centers(self, image, packed, K, D, camera_frame):
        if packed is None:
            return 0, "missing"

        T = self.lookup_static(camera_frame, packed["frame_id"])
        if T is None:
            return 0, "tf"

        count = 0
        for i, obj in enumerate(packed["msg"].objects):
            pose = obj.kinematics.pose_with_covariance.pose
            p = np.array(
                [[pose.position.x, pose.position.y, pose.position.z]],
                dtype=np.float64,
            )
            cam = transform_points(p, T)[0]
            if not np.isfinite(cam).all() or cam[2] <= 0.1:
                continue

            uv, _ = cv2.projectPoints(
                cam.reshape(1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                K,
                D,
            )
            u, v = uv.reshape(2)
            h, w = image.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                x, y = int(round(u)), int(round(v))
                cv2.drawMarker(
                    image, (x, y), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12, thickness=2,
                )
                cv2.putText(
                    image, f"C{i}", (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 255), 1, cv2.LINE_AA,
                )
                count += 1
        return count, "ok"

    def draw_yolox(self, image, packed):
        if packed is None:
            return 0
        count = 0
        try:
            objects = packed["msg"].feature_objects
        except Exception:
            return 0

        for i, item in enumerate(objects):
            roi = item.feature.roi
            x1 = int(roi.x_offset)
            y1 = int(roi.y_offset)
            x2 = int(roi.x_offset + roi.width)
            y2 = int(roi.y_offset + roi.height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                image, f"Y{i}", (x1, max(14, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 220, 0), 1, cv2.LINE_AA,
            )
            count += 1
        return count

    def image_cb(self, msg):
        # Save about 1 debug frame/sec.
        now = time.time()
        if now - self.last_save_wall < 1.0:
            return
        self.last_save_wall = now

        try:
            image = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(str(exc))
            return

        with self.lock:
            if self.K is None or self.camera_frame is None:
                return
            camera_matrix = self.K.copy()
            D = self.D.copy()
            camera_frame = str(self.camera_frame)
            calib_width = int(self.calib_width)
            calib_height = int(self.calib_height)
            raw = self.latest_raw
            fused = self.latest_fused
            cp = self.latest_centerpoint
            yolox = self.latest_yolox

        image_h, image_w = image.shape[:2]

        if calib_width != image_w or calib_height != image_h:
            self.get_logger().error(
                f"CameraInfo/image size mismatch: "
                f"{calib_width}x{calib_height} vs {image_w}x{image_h}"
            )
            return

        # The current CameraInfo is already native 1920x1080. Do not rescale it.
        raw_uv, raw_status = self.project_cloud(
            raw, camera_matrix, D, camera_frame, image.shape, stride=6
        )
        fused_uv, fused_status = self.project_cloud(
            fused, camera_matrix, D, camera_frame, image.shape, stride=30
        )

        # CYAN: raw RSU1/Taki LiDAR. YELLOW: local fused cloud.
        for u, v in raw_uv:
            cv2.circle(image, (int(round(u)), int(round(v))), 1, (255, 255, 0), -1)
        for u, v in fused_uv:
            cv2.circle(image, (int(round(u)), int(round(v))), 1, (0, 255, 255), -1)

        cp_count, cp_status = self.draw_centerpoint_centers(
            image, cp, camera_matrix, D, camera_frame
        )
        y_count = self.draw_yolox(image, yolox)

        def age_ms(packed):
            if packed is None:
                return None
            return (time.time_ns() - packed["arrival_ns"]) * 1e-6

        cv2.putText(
            image,
            "CYAN=raw RSU1 LiDAR  YELLOW=fused cloud  RED=CenterPoint center  GREEN=YOLOX",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            (
                f"raw={len(raw_uv)}({raw_status}) "
                f"fused={len(fused_uv)}({fused_status}) "
                f"centerpoint={cp_count}({cp_status}) yolox={y_count}"
            ),
            (6, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            image,
            (
                f"raw_age={age_ms(raw):.1f}ms " if raw else "raw_age=N/A "
            ) + (
                f"fused_age={age_ms(fused):.1f}ms " if fused else "fused_age=N/A "
            ) + (
                f"cp_age={age_ms(cp):.1f}ms" if cp else "cp_age=N/A"
            ),
            (6, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

        path = self.output_dir / f"compare_{self.frame_index:06d}.jpg"
        cv2.imwrite(str(path), image)

        print(
            f"[SAVE] {path} | "
            f"image={image_w}x{image_h} calib={calib_width}x{calib_height} | "
            f"K fx={camera_matrix[0,0]:.2f} fy={camera_matrix[1,1]:.2f} "
            f"cx={camera_matrix[0,2]:.2f} cy={camera_matrix[1,2]:.2f} | "
            f"raw={len(raw_uv)} fused={len(fused_uv)} cp={cp_count} yolox={y_count}"
        )
        self.frame_index += 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay raw/fused LiDAR, CenterPoint centers, and YOLOX on one camera."
    )
    parser.add_argument("--image-topic", default="/rsu1/gate/image_raw")
    parser.add_argument("--camera-info-topic", default="/rsu1/gate/camera_info")
    parser.add_argument(
        "--raw-pointcloud-topic", default="/rsu1/sensing/lidar/pointcloud"
    )
    parser.add_argument(
        "--fused-pointcloud-topic", default="/sensing/lidar/concatenated/pointcloud"
    )
    parser.add_argument(
        "--centerpoint-topic",
        default="/perception/object_recognition/detection/centerpoint/objects",
    )
    parser.add_argument("--yolox-topic", default="/rsu1/gate/yolox/objects")
    parser.add_argument(
        "--output-dir",
        default="/autoware_runtime/hyelin/results/projection_debug",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ProjectionIsolationDebug(args)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
