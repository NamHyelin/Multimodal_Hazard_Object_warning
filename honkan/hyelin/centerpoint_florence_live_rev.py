#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
import socket
from collections import deque
from pipeline_runtime import (
    InstanceLock,
    create_session_directory,
    make_node_name,
    make_run_id,
)

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from autoware_perception_msgs.msg import DetectedObjects, TrackedObjects
from tf2_ros import Buffer, TransformListener, TransformException



DT_SENDER_VERSION = "UDP_LATEST_ONLY_FINAL_V3"

LABEL_NAMES = {
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


def stamp_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def ros_image_to_bgr(msg: RosImage) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding not in {"bgr8", "rgb8"}:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()

    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def clip_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(1, min(width, round(x2))))
    y2 = int(max(1, min(height, round(y2))))
    return x1, y1, x2, y2


def expand_bbox_with_min_size(
    bbox,
    image_width: int,
    image_height: int,
    expand_ratio: float,
    min_width: int,
    min_height: int,
):
    x1, y1, x2, y2 = bbox
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    target_w = max(box_w * (1.0 + expand_ratio), float(min_width))
    target_h = max(box_h * (1.0 + expand_ratio), float(min_height))

    expanded = (
        cx - target_w / 2.0,
        cy - target_h / 2.0,
        cx + target_w / 2.0,
        cy + target_h / 2.0,
    )
    return clip_bbox(expanded, image_width, image_height)


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion xyzw -> 3x3 rotation matrix."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform) -> np.ndarray:
    """geometry_msgs/Transform -> homogeneous 4x4 matrix."""
    t = transform.translation
    q = transform.rotation

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def make_3d_box_corners(obj) -> np.ndarray:
    """
    Build the 8 corners of the CenterPoint 3D bounding box in the
    CenterPoint message frame.

    shape.dimensions:
      x = box length
      y = box width
      z = box height
    """
    pose = obj.kinematics.pose_with_covariance.pose
    center = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=np.float64,
    )

    length = max(float(obj.shape.dimensions.x), 1e-3)
    width = max(float(obj.shape.dimensions.y), 1e-3)
    height = max(float(obj.shape.dimensions.z), 1e-3)

    hx = length / 2.0
    hy = width / 2.0
    hz = height / 2.0

    local = np.array(
        [
            [ hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy,  hz],
            [ hx, -hy, -hz],
            [-hx,  hy,  hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [-hx, -hy, -hz],
        ],
        dtype=np.float64,
    )

    # Autoware orientation_availability == 0 means orientation is unavailable.
    # In that case, use an axis-aligned box. For 1/2, use the supplied quaternion.
    orientation_availability = int(
        getattr(obj.kinematics, "orientation_availability", 0)
    )

    if orientation_availability == 0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        q = pose.orientation
        rotation = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)

    return (rotation @ local.T).T + center


def position_and_dimensions(obj):
    pose = obj.kinematics.pose_with_covariance.pose
    center = np.array(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=np.float64,
    )
    dims = np.array(
        [
            max(float(obj.shape.dimensions.x), 1e-3),
            max(float(obj.shape.dimensions.y), 1e-3),
            max(float(obj.shape.dimensions.z), 1e-3),
        ],
        dtype=np.float64,
    )
    return center, dims


def dimension_ratio(a: np.ndarray, b: np.ndarray) -> float:
    ratios = np.maximum(a, b) / np.maximum(np.minimum(a, b), 1e-3)
    return float(np.max(ratios))


def load_florence(model_id: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"[Florence] loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"[Florence] loading model: {model_id}")
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)
    model.eval()

    p = next(model.parameters())
    print(f"[Florence] loaded | device={p.device} dtype={p.dtype}")
    return processor, model, dtype


class CenterPointFlorenceNode(Node):
    def __init__(self, args: argparse.Namespace):
        run_id = make_run_id(args.run_id)
        instance_lock = InstanceLock(
            "centerpoint_florence",
            args.image_topic,
            args.camera_info_topic,
            args.pointcloud_topic,
            args.centerpoint_topic,
        )
        super().__init__(make_node_name("centerpoint_florence", run_id))
        self.run_id = run_id
        self.instance_lock = instance_lock

        self.args = args
        self.device = torch.device(args.device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        self.output_dir = create_session_directory(
            args.output_dir,
            "centerpoint_florence",
            run_id,
            {
                "pipeline_role": "CenterPoint 3D ROI -> camera projection -> conditional Florence captioning",
                "stable_identity_output": False,
                "arguments": vars(args),
            },
        )
        self.crop_dir = self.output_dir / "florence_crops"
        self.frame_dir = self.output_dir / "frames"
        self.crop_dir.mkdir()
        self.frame_dir.mkdir()

        self.telemetry_lock = threading.Lock()
        self.create_subscription(TrackedObjects,
            '/perception/object_recognition/tracking/objects',
            self.telemetry_tracking_callback, qos_profile_sensor_data)
        self.result_path = self.output_dir / "results.csv"
        self.result_file = self.result_path.open(
            "x", encoding="utf-8", newline="", buffering=1
        )

        self.csv_fieldnames = [
            "frame_id",
            "camera_timestamp",
            "centerpoint_timestamp",
            "final_timestamp",
            "num_skipped_centerpoint",
            "transmitted_timestamp",
            "object_index",
            "stable_track_id",
            "centerpoint_class",
            "existence_probability",
            "centerpoint_bbox",
            "lidar_position",
            "map_position",
            "florence_caption",
        ]
        self.csv_lock = threading.Lock()
        self.csv_writer = csv.DictWriter(

            self.result_file,
            fieldnames=self.csv_fieldnames,
        )
        self.csv_writer.writeheader()

        # ------------------------------------------------------------------
        # Camera buffer / CameraInfo
        # ------------------------------------------------------------------
        self.image_buffer = deque(maxlen=args.image_buffer_size)
        self.image_lock = threading.Lock()

        self.cloud_buffer = deque(maxlen=args.cloud_buffer_size)
        self.cloud_lock = threading.Lock()

        self.camera_info_lock = threading.Lock()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None
        self.camera_width = None
        self.camera_height = None
        self.intrinsic_scale_logged = False

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )
        self.extrinsic_cache = {}
        self.extrinsic_lock = threading.Lock()

        # ------------------------------------------------------------------
        # Latest-only CenterPoint processing
        # ------------------------------------------------------------------
        self.sensor_callback_group = ReentrantCallbackGroup()
        self.detection_callback_group = MutuallyExclusiveCallbackGroup()

        self.latest_detection_lock = threading.Lock()
        self.latest_detection_event = threading.Event()
        self.stop_worker_event = threading.Event()

        self.latest_detection_msg = None
        self.latest_detection_seq = 0
        self.latest_detection_receive_wall_ns = None
        self.last_processed_detection_seq = 0

        self.frame_index = 0
        self.saved_frame_count = 0
        self.capture_complete = threading.Event()

        # ------------------------------------------------------------------
        # DT-server JSON sender
        #
        # FIFO + retry sender:
        #   - completed perception frames are never overwritten by a newer frame
        #   - one sender thread preserves transmission order
        #   - the same packet is retried before the sender advances to the next one
        #   - HTTP keep-alive is reused when possible
        #
        # NOTE: the perception/Florence worker itself remains latest-only. Thus
        # this guarantees ordered delivery of COMPLETED frames, not every raw
        # CenterPoint callback that arrived while Florence was busy.
        # ------------------------------------------------------------------
        receiver_ip = (args.receiver_ip or "").strip()
        if receiver_ip.lower() == "none":
            receiver_ip = ""

        self.dt_send_enabled = bool(receiver_ip)
        self.dt_receiver_host = receiver_ip
        self.dt_receiver_port = int(args.receiver_port)
        self.dt_receiver_addr = (
            (self.dt_receiver_host, self.dt_receiver_port)
            if self.dt_send_enabled else None
        )

        # One pending DT slot only.
        # If perception finishes several frames while one HTTP POST is in progress,
        # the pending slot is overwritten by the newest completed frame.
        # Therefore DT transmission can never build a FIFO backlog.
        self.dt_pending_lock = threading.Lock()
        self.dt_pending_event = threading.Event()
        self.dt_sender_stop_event = threading.Event()

        self.latest_dt_packet = None
        self.latest_dt_packet_seq = 0
        self.last_attempted_dt_packet_seq = 0
        self.dt_overwritten_pending_count = 0

        self.dt_sender_thread = None
        self.dt_udp_socket = None
        if self.dt_send_enabled:
            self.dt_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Non-blocking UDP: if the local kernel buffer is ever full, drop
            # this stale frame rather than blocking perception.
            self.dt_udp_socket.setblocking(False)

        # ------------------------------------------------------------------
        # 3D semantic cache
        # ------------------------------------------------------------------
        self.semantic_cache = []
        self.next_cache_id = 1

        # ------------------------------------------------------------------
        # Florence
        # ------------------------------------------------------------------
        self.processor, self.florence, self.florence_dtype = load_florence(
            args.florence_model, self.device
        )

        if args.warmup and self.device.type == "cuda":
            self.warmup_florence()

        # ------------------------------------------------------------------
        # ROS subscriptions
        # ------------------------------------------------------------------
        self.create_subscription(
            RosImage,
            args.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )

        self.create_subscription(
            CameraInfo,
            args.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )

        self.create_subscription(
            PointCloud2,
            args.pointcloud_topic,
            self.pointcloud_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )

        self.create_subscription(
            DetectedObjects,
            args.centerpoint_topic,
            self.centerpoint_callback,
            qos_profile_sensor_data,
            callback_group=self.detection_callback_group,
        )

        self.worker = threading.Thread(
            target=self.latest_centerpoint_worker,
            name="latest_centerpoint_worker",
            daemon=True,
        )
        self.worker.start()

        if self.dt_send_enabled:
            self.dt_sender_thread = threading.Thread(
                target=self.dt_sender_worker,
                name="latest_dt_sender",
                daemon=True,
            )
            self.dt_sender_thread.start()

        self.get_logger().info(f"Image topic: {args.image_topic}")
        self.get_logger().info(f"CameraInfo topic: {args.camera_info_topic}")
        self.get_logger().info(f"PointCloud topic: {args.pointcloud_topic}")
        self.get_logger().info(f"CenterPoint topic: {args.centerpoint_topic}")
        self.get_logger().info(
            "Pipeline role: CenterPoint 3D ROI -> camera projection -> "
            "conditional Florence captioning; this node does not perform ID association"
        )
        self.get_logger().warning(
            "object_index is frame-local and must not be treated as a stable track ID"
        )
        self.get_logger().info(
            "CenterPoint processing mode: LATEST-ONLY "
            "(pending older CenterPoint results are overwritten while Florence is busy)"
        )
        self.get_logger().info(
            f"Florence branch: UNKNOWN or existence_probability < "
            f"{args.confidence_threshold:.3f}"
        )
        self.get_logger().info(
            f"3D cache: distance<={args.cache_distance_m:.2f} m, "
            f"dimension-ratio<={args.cache_max_dimension_ratio:.2f}, "
            f"TTL={args.cache_ttl_sec:.1f}s, refresh={args.cache_refresh_sec:.1f}s"
        )
        self.get_logger().info(
            f"2D bbox: projected LiDAR points inside CenterPoint 3D box | "
            f"margin={args.box_point_margin_m:.2f} m | "
            f"sync={args.image_sync_mode}"
        )
        self.get_logger().info(
            f"Florence crop expand={args.crop_expand_ratio:.2f}, "
            f"min={args.min_crop_width}x{args.min_crop_height}"
        )
        self.get_logger().info(f"Run ID: {self.run_id}")
        self.get_logger().info(f"Frame directory: {self.frame_dir}")
        self.get_logger().info(f"Crop directory: {self.crop_dir}")
        self.get_logger().info(f"CSV: {self.result_path}")
        if self.dt_send_enabled:
            self.get_logger().info(
                f"DT SENDER VERSION: {DT_SENDER_VERSION}"
            )
            self.get_logger().info(
                f"DT UDP: {self.dt_receiver_host}:{self.dt_receiver_port} | "
                f"world_frame={args.dt_world_frame} | "
                f"LATEST-ONLY pending slot | no ACK | no retry"
            )
        else:
            self.get_logger().info(
                "DT JSON sender: disabled "
                "(omit --receiver_ip or set --receiver_ip None to disable)"
            )

    def telemetry_tracking_callback(self, msg):
        objects = []
        for obj in msg.objects:
            p = obj.kinematics.pose_with_covariance.pose.position
            v = obj.kinematics.twist_with_covariance.twist.linear
            objects.append(dict(track_id=bytes(obj.object_id.uuid).hex(),
                position=[p.x,p.y,p.z], velocity=[v.x,v.y,v.z],
                classes=[dict(label=int(c.label), probability=float(c.probability))
                         for c in obj.classification]))
        self.telemetry_append('tracking.jsonl', dict(stamp=stamp_sec(msg.header.stamp),
            frame=msg.header.frame_id, received=time.time(), objects=objects))

    def telemetry_append(self, filename, record):
        with self.telemetry_lock:
            with (self.output_dir / filename).open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def close(self):
        self.stop_worker_event.set()
        self.latest_detection_event.set()

        worker = getattr(self, "worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        # No more perception packets can be produced after the worker has ended.
        # Wake the sender so it can transmit the newest pending packet once, then exit.
        self.dt_sender_stop_event.set()
        self.dt_pending_event.set()

        dt_sender = getattr(self, "dt_sender_thread", None)
        if dt_sender is not None and dt_sender.is_alive():
            dt_sender.join(timeout=2.0)

        udp_sock = getattr(self, "dt_udp_socket", None)
        self.dt_udp_socket = None
        if udp_sock is not None:
            try:
                udp_sock.close()
            except Exception:
                pass

        try:
            self.result_file.close()
        except Exception:
            pass
        self.instance_lock.close()

    # ----------------------------------------------------------------------
    # DT-server JSON transmission
    # ----------------------------------------------------------------------
    def transform_position_to_dt_world(
        self,
        position_xyz: np.ndarray,
        source_frame: str,
    ):
        """
        Return (world_position_xyz, world_frame, error).

        By default --dt_world_frame is lidar_fusion, so the CenterPoint position
        is sent unchanged. If the DT server uses another frame (e.g. map), pass
        --dt_world_frame map and TF is used to transform the position.
        """
        target_frame = self.args.dt_world_frame.strip() or source_frame

        if target_frame == source_frame:
            return position_xyz.copy(), source_frame, None

        transform = self.get_extrinsic_matrix(
            target_frame=target_frame,
            source_frame=source_frame,
        )
        if transform is None:
            return None, target_frame, "world_tf_unavailable"

        point_h = np.array(
            [
                float(position_xyz[0]),
                float(position_xyz[1]),
                float(position_xyz[2]),
                1.0,
            ],
            dtype=np.float64,
        )
        transformed = (transform @ point_h)[:3]
        return transformed, target_frame, None

    def build_dt_packet(self, msg: DetectedObjects, frame_record: dict) -> dict:
        """
        Build the compact JSON sent to the DT server.

        The DT receives only the fields needed for freshness, semantic content,
        map/world position, and physical object size.
        """
        objects = []

        for record in frame_record.get("objects", []):
            world_pos = record.get("world_position_xyz")
            dims = record.get("dimensions_xyz")

            obj_packet = {
                "object_id": int(record.get("object_index", 0)),
                "object_id_is_stable": False,
                "source_object_index": int(record.get("object_index", 0)),
                "stable_track_id": None,
                "class": record.get("centerpoint_class"),
                "semantic_source": record.get("semantic_source"),
                "florence_caption": record.get("florence_caption"),

                "world_position_m": (
                    None
                    if world_pos is None
                    else [
                        float(world_pos[0]),
                        float(world_pos[1]),
                        float(world_pos[2]),
                    ]
                ),

                "size_m": (
                    None
                    if dims is None
                    else {
                        "length": float(dims[0]),
                        "width": float(dims[1]),
                        "height": float(dims[2]),
                    }
                ),
            }
            objects.append(obj_packet)

        return {
            "frame_index": int(frame_record.get("frame_index", 0)),
            "latest_only_seq": int(frame_record.get("latest_only_seq", 0)),
            "centerpoint_timestamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "camera_timestamp": frame_record.get("camera_timestamp_sec"),
            "final_timestamp": frame_record.get("result_wall_sec"),
            "objects": objects,
        }

    def enqueue_dt_packet(self, packet: dict, frame_record: dict) -> None:
        """
        Keep only the newest COMPLETED perception frame waiting for DT transport.

        If one HTTP POST is currently in progress, any older waiting packet is
        replaced by the newest packet. This bounds transport backlog to at most
        one pending frame.
        """
        if not self.dt_send_enabled:
            return

        with self.dt_pending_lock:
            had_unsent_pending = (
                self.latest_dt_packet is not None
                and self.latest_dt_packet_seq > self.last_attempted_dt_packet_seq
            )

            self.latest_dt_packet_seq += 1
            slot_seq = self.latest_dt_packet_seq

            if had_unsent_pending:
                self.dt_overwritten_pending_count += 1

            self.latest_dt_packet = {
                "slot_seq": slot_seq,
                "packet": packet,
                "frame_record": frame_record,
                "enqueued_monotonic": time.perf_counter(),
            }

        self.dt_pending_event.set()

        print(
            f"[DT LATEST] frame={packet.get('frame_index'):06d} "
            f"seq={packet.get('latest_only_seq')} "
            f"slot={slot_seq} "
            f"overwritten_pending={self.dt_overwritten_pending_count}",
            flush=True,
        )

    def send_dt_packet_udp(
        self,
        packet: dict,
        frame_record: dict,
        transmitted_timestamp: float,
    ) -> bool:
        """
        Fire-and-forget latest-state UDP transport.

        There is intentionally:
          - no response wait,
          - no retry,
          - no TCP retransmission/head-of-line blocking.

        A lost frame is acceptable because a newer DT frame will replace it.
        """
        packet['run_id'] = self.run_id
        packet['transmitted_timestamp'] = time.time()
        body = json.dumps(
            packet,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        # UDP maximum payload is 65507 bytes. Stay below it explicitly.
        if len(body) > 60000:
            self.get_logger().warning(
                f"[DT UDP DROP OVERSIZE] bytes={len(body)} "
                f"frame={packet.get('frame_index')} "
                f"seq={packet.get('latest_only_seq')}"
            )
            return False

        sock = self.dt_udp_socket
        if sock is None:
            return False

        try:
            sent = sock.sendto(body, self.dt_receiver_addr)
            self.telemetry_append('transmissions.jsonl', dict(
                frame_index=packet['frame_index'], seq=packet['latest_only_seq'],
                tx_wall=packet['transmitted_timestamp'], sent_wall=time.time(),
                bytes=sent))
            if sent != len(body):
                self.get_logger().warning(
                    f"[DT UDP PARTIAL] sent={sent}/{len(body)} "
                    f"frame={packet.get('frame_index')} "
                    f"seq={packet.get('latest_only_seq')}"
                )
                return False

            print(
                f"[DT UDP SEND] frame={packet.get('frame_index'):06d} "
                f"seq={packet.get('latest_only_seq')} "
                f"bytes={sent} "
                f"send_ms={(time.time() - transmitted_timestamp) * 1000.0:.3f}",
                flush=True,
            )
            return True

        except (BlockingIOError, InterruptedError) as exc:
            self.get_logger().warning(
                f"[DT UDP DROP LOCAL_BUSY] {exc} "
                f"frame={packet.get('frame_index')} "
                f"seq={packet.get('latest_only_seq')}"
            )
            return False

        except OSError as exc:
            self.get_logger().warning(
                f"[DT UDP DROP ERROR] {exc} "
                f"frame={packet.get('frame_index')} "
                f"seq={packet.get('latest_only_seq')}"
            )
            return False

    def dt_sender_worker(self) -> None:
        """
        Single-threaded, latest-only DT sender.

        Invariants:
          - at most one packet waits in the pending slot
          - pending packets are overwritten by newer completed frames
          - UDP send never waits for an ACK
          - no retry of stale packets
          - after every send attempt, jump directly to the newest pending frame
        """
        while True:
            self.dt_pending_event.wait(timeout=0.1)

            with self.dt_pending_lock:
                pending = self.latest_dt_packet
                slot_seq = self.latest_dt_packet_seq
                self.dt_pending_event.clear()

            if (
                pending is not None
                and slot_seq > self.last_attempted_dt_packet_seq
            ):
                packet = pending["packet"]
                frame_record = pending["frame_record"]

                transmitted_timestamp = time.time()

                # CSV records only frames that actually reach a transport attempt.
                # Frames overwritten before transmission intentionally have no
                # transmitted_timestamp.
                self.write_csv_frame(
                    frame_record,
                    transmitted_timestamp=transmitted_timestamp,
                )

                self.send_dt_packet_udp(
                    packet,
                    frame_record,
                    transmitted_timestamp=transmitted_timestamp,
                )

                self.last_attempted_dt_packet_seq = slot_seq

            with self.dt_pending_lock:
                newer_pending = (
                    self.latest_dt_packet_seq
                    > self.last_attempted_dt_packet_seq
                )

            if newer_pending:
                # A newer frame arrived while the current UDP send was in progress.
                # Process it immediately, skipping all overwritten intermediates.
                self.dt_pending_event.set()

            if self.dt_sender_stop_event.is_set() and not newer_pending:
                break

        # UDP socket is closed by close(); no HTTP cleanup is needed.

    # ----------------------------------------------------------------------
    # CSV
    # ----------------------------------------------------------------------
    def write_csv_frame(
        self,
        frame_record: dict,
        transmitted_timestamp=None,
    ) -> None:
        """
        Write one CSV block per processed frame.

        Frame-level columns are populated only in the first row of each frame.
        Additional object rows leave those frame-level columns blank.
        """
        frame_base = {
            "frame_id": frame_record.get("frame_index"),
            "camera_timestamp": frame_record.get("camera_timestamp_sec"),
            "centerpoint_timestamp": frame_record.get("stamp_sec"),
            "final_timestamp": frame_record.get("result_wall_sec"),
            "num_skipped_centerpoint": frame_record.get(
                "latest_only_skipped_before"
            ),
            "transmitted_timestamp": transmitted_timestamp,
        }

        objects = frame_record.get("objects", [])

        with self.csv_lock:
            if not objects:
                row = {name: "" for name in self.csv_fieldnames}
                row.update(frame_base)
                self.csv_writer.writerow(row)
                return

            for object_row_index, obj in enumerate(objects):
                row = {name: "" for name in self.csv_fieldnames}

                # Frame-level values only on the first object row.
                if object_row_index == 0:
                    row.update(frame_base)

                bbox = obj.get("image_bbox_px")
                if bbox is None:
                    bbox_text = ""
                else:
                    x1, y1, x2, y2 = bbox
                    bbox_text = f"[{x1},{x2},{y1},{y2}]"

                lidar_pos = obj.get("position_xyz")
                if lidar_pos is None:
                    lidar_text = ""
                else:
                    lidar_text = (
                        f"[{float(lidar_pos[0]):.6f},"
                        f"{float(lidar_pos[1]):.6f},"
                        f"{float(lidar_pos[2]):.6f}]"
                    )

                map_pos = obj.get("world_position_xyz")
                if map_pos is None:
                    map_text = ""
                else:
                    map_text = (
                        f"[{float(map_pos[0]):.6f},"
                        f"{float(map_pos[1]):.6f},"
                        f"{float(map_pos[2]):.6f}]"
                    )

                row.update({
                    "object_index": obj.get("object_index"),
                    "stable_track_id": "",
                    "centerpoint_class": obj.get("centerpoint_class"),
                    "existence_probability": obj.get("existence_probability"),
                    "centerpoint_bbox": bbox_text,
                    "lidar_position": lidar_text,
                    "map_position": map_text,
                    "florence_caption": obj.get("florence_caption") or "",
                })
                self.csv_writer.writerow(row)

    # ----------------------------------------------------------------------
    # Camera
    # ----------------------------------------------------------------------
    def image_callback(self, msg: RosImage) -> None:
        try:
            image = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        item = {
            "stamp_ns": stamp_ns(msg.header.stamp),
            "stamp_sec": stamp_sec(msg.header.stamp),
            "receive_wall_ns": time.time_ns(),
            "frame_id": msg.header.frame_id,
            "image": image,
        }
        with self.image_lock:
            self.image_buffer.append(item)

    def camera_info_callback(self, msg: CameraInfo) -> None:
        k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.asarray(msg.d, dtype=np.float64).reshape(-1)

        with self.camera_info_lock:
            self.camera_matrix = k
            self.dist_coeffs = d
            self.camera_frame = msg.header.frame_id
            self.camera_width = int(msg.width)
            self.camera_height = int(msg.height)

    def projection_camera_matrix(
        self,
        camera_matrix: np.ndarray,
        calibration_width: int,
        calibration_height: int,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Return CameraInfo intrinsics in the image pixel coordinate system.

        Resolution mismatch is rejected by default because a native camera mode
        change is not equivalent to resizing an image.
        """
        K = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3).copy()

        if (
            calibration_width is None
            or calibration_height is None
            or calibration_width <= 0
            or calibration_height <= 0
        ):
            return K

        sx = float(image_width) / float(calibration_width)
        sy = float(image_height) / float(calibration_height)

        if abs(sx - 1.0) < 1e-12 and abs(sy - 1.0) < 1e-12:
            return K

        if not self.args.allow_intrinsic_rescale:
            raise ValueError(
                "CameraInfo/image resolution mismatch: "
                f"calibration={calibration_width}x{calibration_height}, "
                f"image={image_width}x{image_height}. Publish matching native-mode "
                "intrinsics, or pass --allow_intrinsic_rescale only when the image "
                "is a pure resize of the calibrated stream."
            )

        K[0, 0] *= sx   # fx
        K[0, 2] *= sx   # cx
        K[1, 1] *= sy   # fy
        K[1, 2] *= sy   # cy

        if not self.intrinsic_scale_logged:
            self.get_logger().warning(
                "CameraInfo/image resolution mismatch detected. "
                f"Scaling projection intrinsics: "
                f"{calibration_width}x{calibration_height} -> "
                f"{image_width}x{image_height}, "
                f"sx={sx:.6f}, sy={sy:.6f}"
            )
            self.get_logger().info(
                "Projection K after scaling: "
                f"fx={K[0,0]:.3f}, fy={K[1,1]:.3f}, "
                f"cx={K[0,2]:.3f}, cy={K[1,2]:.3f}"
            )
            self.intrinsic_scale_logged = True

        return K

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        item = {
            "stamp": msg.header.stamp,
            "stamp_ns": stamp_ns(msg.header.stamp),
            "stamp_sec": stamp_sec(msg.header.stamp),
            "receive_wall_ns": time.time_ns(),
            "frame_id": msg.header.frame_id,
            "msg": msg,
        }
        with self.cloud_lock:
            self.cloud_buffer.append(item)

    @staticmethod
    def _closest_by_key(candidates, target_value, key):
        if not candidates:
            return None, None, None

        selected = min(
            candidates,
            key=lambda item: abs(item[key] - target_value),
        )
        signed_diff = selected[key] - target_value
        return selected, abs(signed_diff), signed_diff

    def closest_cloud(self, target_stamp_ns: int):
        with self.cloud_lock:
            candidates = list(self.cloud_buffer)

        selected, abs_dt_ns, signed_dt_ns = self._closest_by_key(
            candidates,
            target_stamp_ns,
            "stamp_ns",
        )
        if selected is None:
            return None, None, None

        abs_dt_sec = abs_dt_ns * 1e-9
        signed_dt_sec = signed_dt_ns * 1e-9
        if abs_dt_sec > self.args.max_cloud_time_diff:
            return None, abs_dt_sec, signed_dt_sec

        return selected, abs_dt_sec, signed_dt_sec

    def closest_image_for_cloud(self, cloud_item):
        """
        Alternative synchronization strategy.

        Default mode 'arrival' pairs the camera image and LiDAR cloud using
        receive-wall time inside this process.  This avoids a fixed offset
        between camera and LiDAR header clocks.

        Mode 'timestamp' is also available and supports an explicit
        --camera_time_offset_sec.
        """
        with self.image_lock:
            candidates = list(self.image_buffer)

        if not candidates:
            return None, None, None

        if self.args.image_sync_mode == "arrival":
            selected, abs_dt_ns, signed_dt_ns = self._closest_by_key(
                candidates,
                cloud_item["receive_wall_ns"],
                "receive_wall_ns",
            )
            abs_dt_sec = abs_dt_ns * 1e-9
            signed_dt_sec = signed_dt_ns * 1e-9

            if abs_dt_sec > self.args.max_image_arrival_diff:
                return None, abs_dt_sec, signed_dt_sec

            return selected, abs_dt_sec, signed_dt_sec

        target_stamp_ns = (
            cloud_item["stamp_ns"]
            + int(self.args.camera_time_offset_sec * 1e9)
        )
        selected, abs_dt_ns, signed_dt_ns = self._closest_by_key(
            candidates,
            target_stamp_ns,
            "stamp_ns",
        )
        abs_dt_sec = abs_dt_ns * 1e-9
        signed_dt_sec = signed_dt_ns * 1e-9

        if abs_dt_sec > self.args.max_image_time_diff:
            return None, abs_dt_sec, signed_dt_sec

        return selected, abs_dt_sec, signed_dt_sec

    @staticmethod
    def read_xyz_from_cloud(cloud_msg: PointCloud2) -> np.ndarray:
        """
        Heterogeneous-field-safe PointCloud2 parser.
        """
        pts = point_cloud2.read_points(
            cloud_msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
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


    # ----------------------------------------------------------------------
    # CenterPoint object helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def get_label(obj):
        classifications = obj.classification
        if not classifications:
            return 0, "UNKNOWN", None

        best = max(classifications, key=lambda item: float(item.probability))
        label = int(best.label)
        class_probability = float(best.probability)
        return label, LABEL_NAMES.get(label, f"LABEL_{label}"), class_probability

    @staticmethod
    def inside_road_roi(center: np.ndarray):
        """
        Keep only CenterPoint objects inside the useful road/camera ROI.
        lidar_fusion coordinates:
        Right boundary:
            y >= 0.47 * x - 18.0
        Left boundary:
            y <= 4.0
        """
        x = float(center[0])
        y = float(center[1])

        # Right side: trees / bicycles / outside-road area
        right_boundary_y = 0.47 * x - 18.0

        # Left side: white-building side
        left_boundary_y = 4.0

        if y < right_boundary_y:
            return False, "RIGHT", right_boundary_y, left_boundary_y
        if y > left_boundary_y:
            return False, "LEFT", right_boundary_y, left_boundary_y
        return True, None, right_boundary_y, left_boundary_y

    # ----------------------------------------------------------------------
    # TF and projection
    # ----------------------------------------------------------------------
    def get_extrinsic_matrix(self, target_frame: str, source_frame: str):
        key = (target_frame, source_frame)

        with self.extrinsic_lock:
            cached = self.extrinsic_cache.get(key)
        if cached is not None:
            return cached

        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"TF unavailable: {source_frame} -> {target_frame}: {exc}"
            )
            return None

        matrix = transform_to_matrix(transform_stamped.transform)

        with self.extrinsic_lock:
            self.extrinsic_cache[key] = matrix

        self.get_logger().info(
            f"Cached static extrinsic: {source_frame} -> {target_frame}"
        )
        return matrix

    def lookup_transform_matrix_at(
        self,
        target_frame: str,
        source_frame: str,
        stamp,
    ):
        """
        Use TF at the actual sensor timestamp.  No static/latest TF cache is
        used for camera projection.
        """
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)

        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=self.args.tf_timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Exact-time TF unavailable: "
                f"{source_frame} -> {target_frame}: {exc}"
            )
            return None

        return transform_to_matrix(transform_stamped.transform)

    @staticmethod
    def transform_xyz_points(points_xyz: np.ndarray, matrix: np.ndarray):
        if points_xyz is None or len(points_xyz) == 0:
            return np.empty((0, 3), dtype=np.float64)

        ones = np.ones((len(points_xyz), 1), dtype=np.float64)
        points_h = np.concatenate(
            [points_xyz.astype(np.float64), ones],
            axis=1,
        )
        return (matrix @ points_h.T).T[:, :3]

    def object_points_in_3d_box(
        self,
        cloud_xyz_detection: np.ndarray,
        obj,
    ) -> np.ndarray:
        """
        Select actual LiDAR points belonging to the CenterPoint 3D detection.

        This does NOT use the projected CenterPoint center to make a 2D box.
        Instead, it finds the real 3D LiDAR returns inside the detected object
        volume and later projects those points themselves.
        """
        pose = obj.kinematics.pose_with_covariance.pose
        center = np.array(
            [pose.position.x, pose.position.y, pose.position.z],
            dtype=np.float64,
        )

        dims = np.array(
            [
                max(float(obj.shape.dimensions.x), 1e-3),
                max(float(obj.shape.dimensions.y), 1e-3),
                max(float(obj.shape.dimensions.z), 1e-3),
            ],
            dtype=np.float64,
        )
        half = dims / 2.0
        margin = float(self.args.box_point_margin_m)

        rel = cloud_xyz_detection - center[None, :]

        availability = int(
            getattr(obj.kinematics, "orientation_availability", 0)
        )
        if availability == 0:
            local = rel
        else:
            q = pose.orientation
            rotation = quaternion_to_rotation_matrix(
                q.x, q.y, q.z, q.w
            )
            # row-vector form of local = R^T * rel
            local = rel @ rotation

        xy_ok = (
            (np.abs(local[:, 0]) <= half[0] + margin)
            & (np.abs(local[:, 1]) <= half[1] + margin)
        )

        # Support either geometric-center or bottom-center z conventions.
        z_centered = np.abs(local[:, 2]) <= half[2] + margin
        z_bottom = (
            (local[:, 2] >= -margin)
            & (local[:, 2] <= dims[2] + margin)
        )
        mask = xy_ok & (z_centered | z_bottom)

        return cloud_xyz_detection[mask]

    def project_object_points_to_image(
        self,
        obj,
        cloud_xyz_detection: np.ndarray,
        detection_frame: str,
        camera_frame: str,
        sensor_stamp,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_width: int,
        image_height: int,
    ):
        """
        Build the 2D bbox from projected *actual LiDAR points* inside the
        CenterPoint 3D detection.  No 3D-center-to-2D-center assumption is used.
        """
        object_points = self.object_points_in_3d_box(
            cloud_xyz_detection,
            obj,
        )

        if len(object_points) < self.args.min_object_lidar_points:
            return (
                None,
                f"too_few_object_points:{len(object_points)}",
                len(object_points),
            )

        T_camera_detection = self.lookup_transform_matrix_at(
            target_frame=camera_frame,
            source_frame=detection_frame,
            stamp=sensor_stamp,
        )
        if T_camera_detection is None:
            return None, "tf_unavailable", len(object_points)

        points_camera = self.transform_xyz_points(
            object_points,
            T_camera_detection,
        )

        valid = (
            np.isfinite(points_camera).all(axis=1)
            & (points_camera[:, 2] > self.args.min_camera_depth_m)
        )
        points_camera = points_camera[valid]

        if len(points_camera) < self.args.min_object_lidar_points:
            return (
                None,
                f"too_few_front_points:{len(points_camera)}",
                len(points_camera),
            )

        projected, _ = cv2.projectPoints(
            points_camera.reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            camera_matrix,
            dist_coeffs,
        )
        uv = projected.reshape(-1, 2)

        finite = np.isfinite(uv).all(axis=1)
        uv = uv[finite]
        if len(uv) < self.args.min_object_lidar_points:
            return (
                None,
                f"too_few_projected_points:{len(uv)}",
                len(uv),
            )

        # Robust min/max to suppress a small number of stray points.
        p = float(self.args.bbox_percentile)
        if len(uv) >= 20 and 0.0 < p < 50.0:
            x1 = float(np.percentile(uv[:, 0], p))
            x2 = float(np.percentile(uv[:, 0], 100.0 - p))
            y1 = float(np.percentile(uv[:, 1], p))
            y2 = float(np.percentile(uv[:, 1], 100.0 - p))
        else:
            x1 = float(np.min(uv[:, 0]))
            x2 = float(np.max(uv[:, 0]))
            y1 = float(np.min(uv[:, 1]))
            y2 = float(np.max(uv[:, 1]))

        # Small padding around the LiDAR returns to better cover the visual object.
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        pad = float(self.args.object_bbox_padding_ratio)
        x1 -= 0.5 * pad * w
        x2 += 0.5 * pad * w
        y1 -= 0.5 * pad * h
        y2 += 0.5 * pad * h

        if (
            x2 < 0
            or y2 < 0
            or x1 >= image_width
            or y1 >= image_height
        ):
            return None, "outside_image", len(uv)

        # Clip ONLY in the actual received image coordinate system
        # (e.g. 1920x1080), never in CameraInfo's 640x480 metadata size.
        bbox = clip_bbox(
            (x1, y1, x2, y2),
            image_width,
            image_height,
        )

        if bbox[2] - bbox[0] <= 1 or bbox[3] - bbox[1] <= 1:
            return None, "tiny_projected_bbox", len(uv)

        return bbox, None, len(uv)

    # ----------------------------------------------------------------------
    # 3D semantic cache
    # ----------------------------------------------------------------------
    @staticmethod
    def cache_class_compatible(current_label: int, cached_label: int) -> bool:
        return (
            current_label == cached_label
            or current_label == 0
            or cached_label == 0
        )

    def purge_semantic_cache(self, now_ns: int) -> None:
        ttl_ns = int(self.args.cache_ttl_sec * 1e9)
        self.semantic_cache = [
            entry
            for entry in self.semantic_cache
            if now_ns - entry["last_seen_ns"] <= ttl_ns
        ]

    def find_cache_match(
        self,
        label: int,
        center: np.ndarray,
        dims: np.ndarray,
        now_ns: int,
        used_cache_ids: set,
    ):
        self.purge_semantic_cache(now_ns)

        best = None
        best_distance = float("inf")

        for entry in self.semantic_cache:
            if entry["cache_id"] in used_cache_ids:
                continue

            if not self.cache_class_compatible(label, entry["label"]):
                continue

            distance_m = float(np.linalg.norm(center - entry["center"]))
            dims_ratio = dimension_ratio(dims, entry["dims"])

            if distance_m > self.args.cache_distance_m:
                continue
            if dims_ratio > self.args.cache_max_dimension_ratio:
                continue

            if distance_m < best_distance:
                best_distance = distance_m
                best = {
                    "entry": entry,
                    "distance_m": distance_m,
                    "dimension_ratio": dims_ratio,
                }

        return best

    def update_cache_track(
        self,
        entry,
        label: int,
        center: np.ndarray,
        dims: np.ndarray,
        now_ns: int,
    ) -> None:
        if label != 0:
            entry["label"] = label

        entry["center"] = center.copy()
        entry["dims"] = dims.copy()
        entry["last_seen_ns"] = now_ns
        entry["last_seen_frame"] = self.frame_index

    def create_cache_entry(
        self,
        label: int,
        center: np.ndarray,
        dims: np.ndarray,
        caption: str,
        now_ns: int,
    ):
        entry = {
            "cache_id": self.next_cache_id,
            "label": label,
            "center": center.copy(),
            "dims": dims.copy(),
            "caption": caption,
            "caption_created_ns": now_ns,
            "last_seen_ns": now_ns,
            "last_seen_frame": self.frame_index,
        }
        self.next_cache_id += 1
        self.semantic_cache.append(entry)
        return entry

    # ----------------------------------------------------------------------
    # Florence
    # ----------------------------------------------------------------------
    def prepare_florence_inputs(self, crop_bgr: np.ndarray):
        start = time.perf_counter()

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        inputs = self.processor(
            text=self.args.florence_prompt,
            images=image,
            return_tensors="pt",
        )
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(
            device=self.device,
            dtype=self.florence_dtype,
        )
        sync_cuda(self.device)
        h2d_ms = (time.perf_counter() - start) * 1000.0

        return input_ids, pixel_values, preprocess_ms, h2d_ms

    def warmup_florence(self) -> None:
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        input_ids, pixel_values, _, _ = self.prepare_florence_inputs(dummy)

        sync_cuda(self.device)
        start = time.perf_counter()

        with torch.inference_mode():
            _ = self.florence.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=2,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )

        sync_cuda(self.device)
        self.get_logger().info(
            f"Florence warm-up: {(time.perf_counter() - start) * 1000.0:.1f} ms"
        )

    def run_florence(self, crop_bgr: np.ndarray):
        total_start = time.perf_counter()

        input_ids, pixel_values, preprocess_ms, h2d_ms = (
            self.prepare_florence_inputs(crop_bgr)
        )

        sync_cuda(self.device)
        generate_start = time.perf_counter()

        with torch.inference_mode():
            generated_ids = self.florence.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=self.args.florence_max_new_tokens,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )

        sync_cuda(self.device)
        generate_ms = (time.perf_counter() - generate_start) * 1000.0

        decode_start = time.perf_counter()
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timings = {
            "preprocess_ms": preprocess_ms,
            "h2d_ms": h2d_ms,
            "generate_ms": generate_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
        }
        return caption or "unknown", timings

    @staticmethod
    def draw_bbox(image, bbox, text, color, thickness=2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text_y = max(18, y1 - 6)
        cv2.putText(
            image,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------------------
    # Latest-only callbacks
    # ----------------------------------------------------------------------
    def centerpoint_callback(self, msg: DetectedObjects) -> None:
        if self.capture_complete.is_set():
            return

        receive_wall_ns = time.time_ns()

        with self.latest_detection_lock:
            self.latest_detection_seq += 1
            self.latest_detection_msg = msg
            self.latest_detection_receive_wall_ns = receive_wall_ns

        self.latest_detection_event.set()

    def latest_centerpoint_worker(self) -> None:
        while not self.stop_worker_event.is_set():
            self.latest_detection_event.wait(timeout=0.1)

            if self.stop_worker_event.is_set() or self.capture_complete.is_set():
                break

            with self.latest_detection_lock:
                msg = self.latest_detection_msg
                seq = self.latest_detection_seq
                receive_wall_ns = self.latest_detection_receive_wall_ns
                self.latest_detection_event.clear()

            if msg is None or seq <= self.last_processed_detection_seq:
                continue

            skipped_before = max(
                0,
                seq - self.last_processed_detection_seq - 1,
            )

            processing_start_wall_ns = time.time_ns()

            try:
                self.process_centerpoint_message(
                    msg=msg,
                    latest_only_seq=seq,
                    skipped_before=skipped_before,
                    receive_wall_ns=receive_wall_ns,
                    processing_start_wall_ns=processing_start_wall_ns,
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Latest-only CenterPoint worker error: {exc}"
                )
            finally:
                self.last_processed_detection_seq = seq

            with self.latest_detection_lock:
                if self.latest_detection_seq > self.last_processed_detection_seq:
                    self.latest_detection_event.set()

    # ----------------------------------------------------------------------
    # Main CenterPoint -> camera -> Florence pipeline
    # ----------------------------------------------------------------------
    def process_centerpoint_message(
        self,
        msg: DetectedObjects,
        latest_only_seq: int,
        skipped_before: int,
        receive_wall_ns: int,
        processing_start_wall_ns: int,
    ) -> None:
        callback_start = time.perf_counter()

        centerpoint_stamp_ns = stamp_ns(msg.header.stamp)

        selected_cloud, cloud_dt_sec, cloud_signed_dt_sec = self.closest_cloud(
            centerpoint_stamp_ns
        )

        if selected_cloud is None:
            print(
                f"[FRAME {self.frame_index:06d}] SKIP: "
                f"No matching CenterPoint input cloud within "
                f"{self.args.max_cloud_time_diff * 1000.0:.1f} ms "
                f"(closest={'N/A' if cloud_dt_sec is None else f'{cloud_dt_sec * 1000.0:.1f} ms'})"
            )
            self.frame_index += 1
            return

        selected_image, image_dt_sec, image_signed_dt_sec = (
            self.closest_image_for_cloud(selected_cloud)
        )

        frame_record = {
            "frame_index": self.frame_index,
            "stamp_sec": stamp_sec(msg.header.stamp),
            "camera_timestamp_sec": (
                None if selected_image is None else selected_image["stamp_sec"]
            ),
            "latest_only_seq": latest_only_seq,
            "latest_only_skipped_before": skipped_before,
            "result_wall_sec": None,
            "result_age_from_centerpoint_ms": None,
            "objects": [],
            "receive_wall_ns": receive_wall_ns,
            "processing_start_wall_ns": processing_start_wall_ns,
            "frame_path": None,
        }

        print()
        print("=" * 115)
        print(
            f"[FRAME {self.frame_index:06d}] "
            f"CenterPoint timestamp={int(msg.header.stamp.sec)}."
            f"{int(msg.header.stamp.nanosec):09d} | "
            f"frame={msg.header.frame_id} | "
            f"objects={len(msg.objects)} | "
            f"latest_seq={latest_only_seq} skipped={skipped_before} | "
            f"cloud-centerpoint="
            f"{cloud_signed_dt_sec * 1000.0:+.3f} ms | "
            f"image-cloud({self.args.image_sync_mode})="
            f"{'N/A' if image_signed_dt_sec is None else f'{image_signed_dt_sec * 1000.0:+.3f} ms'}"
        )

        if selected_image is None:
            result_wall_ns = time.time_ns()
            frame_record["result_wall_sec"] = result_wall_ns * 1e-9
            frame_record["result_age_from_centerpoint_ms"] = (
                result_wall_ns - centerpoint_stamp_ns
            ) * 1e-6
            # Keep the skipped frame in CSV. No transmission occurs, so
            # transmitted_timestamp is blank.
            self.write_csv_frame(
                frame_record,
                transmitted_timestamp=None,
            )
            print(
                f"[FRAME {self.frame_index:06d}] SKIP: "
                f"No camera image matched by "
                f"{self.args.image_sync_mode} synchronization."
            )

            self.frame_index += 1
            return

        image_bgr = selected_image["image"]
        image_height, image_width = image_bgr.shape[:2]
        annotated = image_bgr.copy()

        with self.camera_info_lock:
            if self.camera_matrix is None or self.camera_frame is None:
                self.get_logger().warning("CameraInfo unavailable.")
                self.frame_index += 1
                return
            camera_matrix_native = self.camera_matrix.copy()
            dist_coeffs = self.dist_coeffs.copy()
            camera_frame = str(self.camera_frame)
            calibration_width = int(self.camera_width)
            calibration_height = int(self.camera_height)

        # Project using intrinsics for the actual image mode. A resolution mismatch
        # is rejected unless pure resize scaling was explicitly enabled.
        camera_matrix = self.projection_camera_matrix(
            camera_matrix=camera_matrix_native,
            calibration_width=calibration_width,
            calibration_height=calibration_height,
            image_width=image_width,
            image_height=image_height,
        )

        try:
            cloud_xyz = self.read_xyz_from_cloud(selected_cloud["msg"])
        except Exception as exc:
            self.get_logger().error(f"PointCloud2 parse failed: {exc}")
            self.frame_index += 1
            return

        # Put raw cloud points into the same frame as CenterPoint objects.
        if selected_cloud["frame_id"] == msg.header.frame_id:
            cloud_xyz_detection = cloud_xyz
        else:
            T_detection_cloud = self.lookup_transform_matrix_at(
                target_frame=msg.header.frame_id,
                source_frame=selected_cloud["frame_id"],
                stamp=selected_cloud["msg"].header.stamp,
            )
            if T_detection_cloud is None:
                self.frame_index += 1
                return
            cloud_xyz_detection = self.transform_xyz_points(
                cloud_xyz,
                T_detection_cloud,
            )

        used_cache_ids = set()

        for object_index, obj in enumerate(msg.objects):
            confidence = float(obj.existence_probability)
            label, class_name, class_probability = self.get_label(obj)
            center, dims = position_and_dimensions(obj)

            # ----------------------------------------------------------
            # LiDAR-space road ROI filter
            #
            # keep:
            #     0.47*x - 18 <= y <= 4.0
            #
            # LEFT  -> white building side
            # RIGHT -> trees / bicycles side
            # ----------------------------------------------------------
            (
                inside_road,
                outside_side,
                right_boundary_y,
                left_boundary_y,
            ) = self.inside_road_roi(center)

            if not inside_road:
                print(
                    f"  [{object_index:02d}] OUTSIDE {outside_side} ROAD "
                    f"class={class_name:<12} "
                    f"conf={confidence:.3f} "
                    f"lidar=({center[0]:.2f},"
                    f"{center[1]:.2f},"
                    f"{center[2]:.2f}) "
                    f"right_y={right_boundary_y:.2f} "
                    f"left_y={left_boundary_y:.2f}"
                )
                continue

            world_center, world_frame, world_position_error = (
                self.transform_position_to_dt_world(
                    center,
                    source_frame=msg.header.frame_id,
                )
            )

            pose = obj.kinematics.pose_with_covariance.pose
            orientation_xyzw = [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
            orientation_availability = int(
                getattr(obj.kinematics, "orientation_availability", 0)
            )

            uncertain = (
                label == 0 or confidence < self.args.confidence_threshold
            )

            image_bbox, projection_error, projected_point_count = (
                self.project_object_points_to_image(
                    obj=obj,
                    cloud_xyz_detection=cloud_xyz_detection,
                    detection_frame=msg.header.frame_id,
                    camera_frame=camera_frame,
                    sensor_stamp=selected_cloud["msg"].header.stamp,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
            camera_position = None

            record = {
                "object_index": object_index,
                "centerpoint_label": label,
                "centerpoint_class": class_name,
                "classification_probability": class_probability,
                "existence_probability": confidence,
                "source_frame": msg.header.frame_id,
                "position_xyz": center.tolist(),
                "world_frame": world_frame,
                "world_position_xyz": (
                    None if world_center is None else world_center.tolist()
                ),
                "world_position_error": world_position_error,
                "camera_position_xyz": camera_position,
                "projected_lidar_point_count": projected_point_count,
                "dimensions_xyz": dims.tolist(),
                "orientation_xyzw": orientation_xyzw,
                "orientation_availability": orientation_availability,
                "image_bbox_px": (
                    None if image_bbox is None else list(image_bbox)
                ),
                "florence_caption": None,
                "florence_triggered": False,
                "semantic_source": "CenterPoint",
                "cache_id": None,
                "cache_hit": False,
                "projection_error": projection_error,
                "error": None,
            }

            cache_match = self.find_cache_match(
                label=label,
                center=center,
                dims=dims,
                now_ns=centerpoint_stamp_ns,
                used_cache_ids=used_cache_ids,
            )

            if cache_match is not None:
                cache_entry = cache_match["entry"]
                self.update_cache_track(
                    cache_entry,
                    label=label,
                    center=center,
                    dims=dims,
                    now_ns=centerpoint_stamp_ns,
                )
                used_cache_ids.add(cache_entry["cache_id"])
                record["cache_id"] = cache_entry["cache_id"]

            # CenterPoint detects 360 degrees. Objects outside this camera's FOV
            # cannot be cropped or drawn.
            if image_bbox is None:
                print(
                    f"  [{object_index:02d}] NOT VISIBLE "
                    f"class={class_name:<12} conf={confidence:.3f} "
                    f"pos=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}) "
                    f"reason={projection_error}"
                )
                frame_record["objects"].append(record)
                continue

            if not uncertain:
                self.draw_bbox(
                    annotated,
                    image_bbox,
                    f"{class_name} {confidence:.2f}",
                    (0, 220, 0),
                    thickness=2,
                )

                map_text = (
                    "N/A"
                    if world_center is None
                    else (
                        f"({world_center[0]:.2f},"
                        f"{world_center[1]:.2f},"
                        f"{world_center[2]:.2f})"
                    )
                )
                print(
                    f"  [{object_index:02d}] CENTERPOINT "
                    f"class={class_name:<12} conf={confidence:.3f} "
                    f"bbox={image_bbox} points={projected_point_count} | "
                    f"lidar=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
                    f"map={map_text}"
                )
                frame_record["objects"].append(record)
                continue

            # Keep the CenterPoint-derived object bbox tight, and make a
            # separate larger crop only for Florence.
            crop_bbox = expand_bbox_with_min_size(
                image_bbox,
                image_width=image_width,
                image_height=image_height,
                expand_ratio=self.args.crop_expand_ratio,
                min_width=self.args.min_crop_width,
                min_height=self.args.min_crop_height,
            )

            self.draw_bbox(
                annotated,
                image_bbox,
                f"LOW {class_name} {confidence:.2f}",
                (0, 165, 255),
                thickness=2,
            )
            self.draw_bbox(
                annotated,
                crop_bbox,
                "I2T CROP",
                (255, 0, 255),
                thickness=2,
            )

            # Reuse previous Florence caption for the same 3D object.
            if cache_match is not None:
                cache_entry = cache_match["entry"]
                caption_age_ms = (
                    centerpoint_stamp_ns - cache_entry["caption_created_ns"]
                ) * 1e-6

                if caption_age_ms <= self.args.cache_refresh_sec * 1000.0:
                    caption = cache_entry["caption"]

                    record["cache_hit"] = True
                    record["semantic_source"] = "FlorenceCache"
                    record["florence_caption"] = caption

                    self.draw_bbox(
                        annotated,
                        crop_bbox,
                        f"CACHE: {caption[:42]}",
                        (255, 255, 0),
                        thickness=2,
                    )

                    print(
                        f"  [{object_index:02d}] CACHE REUSE "
                        f"id={cache_entry['cache_id']} "
                        f"class={class_name:<12} conf={confidence:.3f} | "
                        f"3D_dist={cache_match['distance_m']:.2f} m "
                        f"dim_ratio={cache_match['dimension_ratio']:.2f} | "
                        f"caption={caption!r}"
                    )

                    frame_record["objects"].append(record)
                    continue

            x1, y1, x2, y2 = crop_bbox
            crop = image_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                record["error"] = "Florence crop is empty."
                frame_record["objects"].append(record)
                continue

            try:
                caption, timings = self.run_florence(crop)
                record['i2t_timings_ms'] = timings
                record['crop_shape'] = list(crop.shape)
                record['caption_utf8_bytes'] = len(caption.encode('utf-8'))

                record["florence_triggered"] = True
                record["semantic_source"] = "Florence"
                record["florence_caption"] = caption

                if cache_match is not None:
                    cache_entry = cache_match["entry"]
                    cache_entry["caption"] = caption
                    cache_entry["caption_created_ns"] = centerpoint_stamp_ns
                    self.update_cache_track(
                        cache_entry,
                        label=label,
                        center=center,
                        dims=dims,
                        now_ns=centerpoint_stamp_ns,
                    )
                else:
                    cache_entry = self.create_cache_entry(
                        label=label,
                        center=center,
                        dims=dims,
                        caption=caption,
                        now_ns=centerpoint_stamp_ns,
                    )
                    used_cache_ids.add(cache_entry["cache_id"])

                record["cache_id"] = cache_entry["cache_id"]

                self.draw_bbox(
                    annotated,
                    crop_bbox,
                    f"I2T: {caption[:45]}",
                    (255, 0, 255),
                    thickness=2,
                )

                map_text = (
                    "N/A"
                    if world_center is None
                    else (
                        f"({world_center[0]:.2f},"
                        f"{world_center[1]:.2f},"
                        f"{world_center[2]:.2f})"
                    )
                )
                print(
                    f"  [{object_index:02d}] FLORENCE "
                    f"class={class_name:<12} conf={confidence:.3f} "
                    f"bbox={image_bbox} crop={crop_bbox} "
                    f"points={projected_point_count} | "
                    f"lidar=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
                    f"map={map_text} | "
                    f"caption={caption!r} | "
                    f"generate={timings['generate_ms']:.1f} ms "
                    f"total={timings['total_ms']:.1f} ms"
                )

                if self.args.save_crops:
                    crop_path = self.crop_dir / (
                        f"frame_{self.frame_index:06d}_"
                        f"centerpoint_{int(msg.header.stamp.sec)}_"
                        f"{int(msg.header.stamp.nanosec):09d}_"
                        f"object_{object_index:02d}_"
                        f"{class_name.lower()}_{confidence:.3f}.jpg"
                    )

                    cv2.imwrite(
                        str(crop_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
                    )

            except Exception as exc:
                record["error"] = str(exc)
                print(
                    f"  [{object_index:02d}] FLORENCE ERROR: {exc}"
                )

            frame_record["objects"].append(record)

        # ------------------------------------------------------------------
        # Save annotated frame
        # ------------------------------------------------------------------
        cv2.putText(
            annotated,
            (
                f"CenterPoint frame {self.frame_index:06d} | "
                f"point-cluster projection | "
                f"img-cloud={image_signed_dt_sec * 1000.0:+.1f} ms"
            ),
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame_path = self.frame_dir / (
            f"frame_{self.frame_index:06d}_"
            f"centerpoint_{int(msg.header.stamp.sec)}_"
            f"{int(msg.header.stamp.nanosec):09d}.jpg"
        )

        saved = cv2.imwrite(
            str(frame_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        if not saved:
            self.get_logger().error(f"Failed to save annotated frame: {frame_path}")
            return
        self.saved_frame_count += 1
        frame_record["frame_path"] = str(frame_path)

        # Final timestamp is taken after processing + frame save.
        result_wall_ns = time.time_ns()
        frame_record["result_wall_sec"] = result_wall_ns * 1e-9
        frame_record["result_age_from_centerpoint_ms"] = (
            result_wall_ns - centerpoint_stamp_ns
        ) * 1e-6

        # With DT transmission enabled, CSV is written by the sender at the
        # actual HTTP POST start so transmitted_timestamp is exact.
        # With transmission disabled, still save the processed frame to CSV
        # and leave transmitted_timestamp blank.
        if self.dt_send_enabled:
            self.enqueue_dt_packet(
                self.build_dt_packet(msg, frame_record),
                frame_record,
            )
        else:
            self.write_csv_frame(
                frame_record,
                transmitted_timestamp=None,
            )

        pipeline_ms = (time.perf_counter() - callback_start) * 1000.0
        print(
            f"[FRAME {self.frame_index:06d}] "
            f"pipeline={pipeline_ms:.1f} ms | "
            f"result age="
            f"{frame_record['result_age_from_centerpoint_ms']:.1f} ms | "
            f"semantic cache={len(self.semantic_cache)}"
        )

        self.telemetry_append('frames.jsonl', frame_record)
        self.frame_index += 1
        if (
            self.args.max_saved_frames > 0
            and self.saved_frame_count >= self.args.max_saved_frames
        ):
            self.get_logger().info(
                f"Saved {self.saved_frame_count} frames; capture complete."
            )
            self.capture_complete.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CenterPoint 3D box -> select actual LiDAR returns -> project point cluster -> "
            "uncertainty-triggered Florence semantic refinement."
        )
    )

    parser.add_argument(
        "--image_topic",
        default="/rsu1/gate/image_raw",
    )
    parser.add_argument(
        "--camera_info_topic",
        default="/rsu1/gate/camera_info",
    )
    parser.add_argument(
        "--centerpoint_topic",
        default="/perception/object_recognition/detection/centerpoint/objects",
    )
    parser.add_argument(
        "--pointcloud_topic",
        default="/sensing/lidar/concatenated/pointcloud",
    )

    parser.add_argument(
        "--florence_model",
        default="florence-community/Florence-2-base-ft",
    )
    parser.add_argument(
        "--florence_prompt",
        default="Describe what is shown in the image with heading direction.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.4,
    )

    # Camera / projection
    parser.add_argument(
        "--max_image_time_diff",
        type=float,
        default=0.10,
        help="Maximum CenterPoint-vs-camera timestamp difference [s].",
    )
    parser.add_argument(
        "--image_buffer_size",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--cloud_buffer_size",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max_cloud_time_diff",
        type=float,
        default=0.10,
        help="Maximum CenterPoint-vs-input-cloud timestamp difference [s].",
    )
    parser.add_argument(
        "--image_sync_mode",
        choices=["arrival", "timestamp"],
        default="arrival",
        help=(
            "arrival: pair camera/cloud using receive wall time; "
            "timestamp: pair using message header timestamps."
        ),
    )
    parser.add_argument(
        "--max_image_arrival_diff",
        type=float,
        default=0.25,
        help="Maximum receive-wall-time difference for arrival sync [s].",
    )
    parser.add_argument(
        "--camera_time_offset_sec",
        type=float,
        default=0.0,
        help=(
            "Only for --image_sync_mode timestamp. "
            "Target camera stamp = cloud stamp + this offset."
        ),
    )
    parser.add_argument(
        "--allow_intrinsic_rescale",
        action="store_true",
        help="Allow K scaling on size mismatch; only for a pure image resize.",
    )
    parser.add_argument(
        "--tf_timeout_sec",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--min_camera_depth_m",
        type=float,
        default=0.10,
    )

    # LiDAR-point cluster projection
    parser.add_argument(
        "--box_point_margin_m",
        type=float,
        default=0.25,
        help="3D margin around CenterPoint box when selecting LiDAR points.",
    )
    parser.add_argument(
        "--min_object_lidar_points",
        type=int,
        default=3,
        help="Minimum selected/projected LiDAR points required for a 2D bbox.",
    )
    parser.add_argument(
        "--bbox_percentile",
        type=float,
        default=2.0,
        help="Robust lower/upper percentile for projected-point bbox.",
    )
    parser.add_argument(
        "--object_bbox_padding_ratio",
        type=float,
        default=0.20,
        help="Small padding around the projected LiDAR-point bbox.",
    )

    # Florence crop
    parser.add_argument(
        "--crop_expand_ratio",
        type=float,
        default=0.5,
        help="Legacy argument; center+depth crop uses --center_crop_expand_ratio.",
    )
    parser.add_argument(
        "--center_crop_expand_ratio",
        type=float,
        default=1.6,
        help=(
            "Deprecated compatibility option. Object bbox is now tight; "
            "Florence expansion uses --crop_expand_ratio."
        ),
    )
    parser.add_argument(
        "--center_visibility_margin_px",
        type=float,
        default=40.0,
        help="Allow projected object centers this many pixels outside the image.",
    )
    parser.add_argument(
        "--min_crop_width",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--min_crop_height",
        type=int,
        default=64,
    )

    # 3D cache
    parser.add_argument(
        "--cache_distance_m",
        type=float,
        default=1.5,
        help="Maximum 3D center displacement to reuse a semantic cache entry.",
    )
    parser.add_argument(
        "--cache_max_dimension_ratio",
        type=float,
        default=2.5,
        help="Maximum box-dimension ratio for cache association.",
    )
    parser.add_argument(
        "--cache_ttl_sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--cache_refresh_sec",
        type=float,
        default=5.0,
    )

    # Florence
    parser.add_argument(
        "--florence_max_new_tokens",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--output_dir",
        default="/autoware_runtime/hyelin/results",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Unique result-session name; defaults to a JST timestamp plus PID.",
    )
    parser.add_argument(
        "--max_saved_frames",
        type=int,
        default=0,
        help="Stop after saving this many annotated frames; 0 means unlimited.",
    )

    # DT server JSON transmission.
    # Omit --receiver_ip, or pass --receiver_ip None, to disable transmission.
    parser.add_argument(
        "--receiver_ip",
        default=None,
        help=(
            "DT server IPv4/hostname. Example: 192.168.1.50. "
            "Omit this option or pass 'None' to disable DT transmission."
        ),
    )
    parser.add_argument(
        "--receiver_port",
        type=int,
        default=8001,
        help="DT server HTTP port.",
    )
    parser.add_argument(
        "--receiver_path",
        default="/dt/update",
        help="DT server HTTP endpoint path.",
    )
    parser.add_argument(
        "--dt_world_frame",
        default="map",
        help=(
            "Coordinate frame used for map_position in CSV and "
            "world_position_m in the DT JSON. Default: map."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_saved_frames < 0:
        raise ValueError("--max_saved_frames must be zero or greater")

    rclpy.init()
    node = CenterPointFlorenceNode(args)

    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    try:
        while rclpy.ok() and not node.capture_complete.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
