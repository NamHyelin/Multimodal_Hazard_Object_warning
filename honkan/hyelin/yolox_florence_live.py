#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image as RosImage
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from pipeline_runtime import (
    InstanceLock,
    create_session_directory,
    make_node_name,
    make_run_id,
)
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



def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    inter = iw * ih

    area_a = max(1.0, float((ax2 - ax1) * (ay2 - ay1)))
    area_b = max(1.0, float((bx2 - bx1) * (by2 - by1)))
    union = area_a + area_b - inter

    return 0.0 if union <= 0.0 else inter / union


def bbox_center_distance(a, b) -> float:
    acx, acy = bbox_center(a)
    bcx, bcy = bbox_center(b)
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def bbox_area_ratio(a, b) -> float:
    area_a = max(1.0, float((a[2] - a[0]) * (a[3] - a[1])))
    area_b = max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))
    return max(area_a, area_b) / min(area_a, area_b)


def expand_bbox_with_min_size(
    bbox,
    image_width: int,
    image_height: int,
    expand_ratio: float,
    min_width: int,
    min_height: int,
):
    """Expand the YOLO bbox and guarantee a minimum Florence crop size."""
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


class YoloxFlorenceNode(Node):
    def __init__(self, args: argparse.Namespace):
        run_id = make_run_id(args.run_id)
        instance_lock = InstanceLock(
            "yolo_florence",
            args.image_topic,
            args.yolox_topic,
        )
        super().__init__(make_node_name("yolo_florence", run_id))
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

        self.output_dir = create_session_directory(
            args.output_dir,
            "yolo_florence",
            run_id,
            {
                "pipeline_role": "2D detector ROI -> conditional Florence captioning",
                "stable_identity_output": False,
                "arguments": vars(args),
            },
        )
        self.crop_dir = self.output_dir / "florence_crops"
        self.frame_dir = self.output_dir / "frames"
        self.crop_dir.mkdir()
        self.frame_dir.mkdir()

        self.result_path = self.output_dir / "results.csv"
        self.result_file = self.result_path.open(
            "x", encoding="utf-8", newline="", buffering=1
        )

        self.csv_fieldnames = [
            "camera_timestamp_sec",
            "yolo_timestamp_sec",
            "final_timestamp",
            "num_skipped_YOLO",
            "process_time_ms_afterYOLO",
            "yolox_class",
            "existence_probability",
            "yolo_bbox",
            "florence_caption",
        ]

        self.csv_writer = csv.DictWriter(
            self.result_file,
            fieldnames=self.csv_fieldnames,
        )
        self.csv_writer.writeheader()

        self.image_buffer = deque(maxlen=args.image_buffer_size)
        self.image_lock = threading.Lock()

        # Camera keeps updating while Florence is running.
        self.sensor_callback_group = ReentrantCallbackGroup()

        # YOLO callbacks are intentionally lightweight: they only update the
        # latest-message slot. Serializing them keeps that update deterministic.
        self.detection_callback_group = MutuallyExclusiveCallbackGroup()

        self.frame_index = 0

        # ------------------------------------------------------------------
        # Latest-only YOLO processing
        #
        # YOLO callbacks never run Florence directly. They only overwrite the
        # single latest message slot. A dedicated worker processes one message
        # at a time. While Florence is busy, newer YOLO results replace older
        # pending results, so once the current job finishes the worker jumps to
        # the most recent YOLO result instead of draining an old backlog.
        # ------------------------------------------------------------------
        self.latest_yolo_lock = threading.Lock()
        self.latest_yolo_event = threading.Event()
        self.stop_worker_event = threading.Event()

        self.latest_yolo_msg = None
        self.latest_yolo_seq = 0
        self.latest_yolo_receive_wall_ns = None
        self.last_processed_yolo_seq = 0

        # Semantic cache:
        # once Florence describes a low-confidence object, remember the caption
        # and its latest image-space bbox. Subsequent detections at a similar
        # position reuse the caption instead of launching Florence again.
        self.semantic_cache = []
        self.next_cache_id = 1

        self.processor, self.florence, self.florence_dtype = load_florence(
            args.florence_model, self.device
        )

        if args.warmup and self.device.type == "cuda":
            self.warmup_florence()

        self.create_subscription(
            RosImage,
            args.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_callback_group,
        )

        self.create_subscription(
            DetectedObjectsWithFeature,
            args.yolox_topic,
            self.yolox_callback,
            qos_profile_sensor_data,
            callback_group=self.detection_callback_group,
        )

        self.yolo_worker = threading.Thread(
            target=self.latest_yolo_worker,
            name="latest_yolo_worker",
            daemon=True,
        )
        self.yolo_worker.start()

        self.get_logger().info(f"Image topic: {args.image_topic}")
        self.get_logger().info(f"Detector topic: {args.yolox_topic}")
        self.get_logger().info(
            "Pipeline role: detector ROI -> conditional Florence captioning; "
            "this node does not perform ID association"
        )
        self.get_logger().info(
            "YOLO processing mode: LATEST-ONLY "
            "(pending older YOLO results are overwritten while Florence is busy)"
        )
        self.get_logger().info(
            f"Florence branch: UNKNOWN or existence_probability < "
            f"{args.confidence_threshold:.3f}"
        )
        self.get_logger().info(
            f"Florence crop: expand={args.crop_expand_ratio:.2f}, "
            f"min={args.min_crop_width}x{args.min_crop_height}"
        )
        self.get_logger().info(
            "Semantic cache: "
            f"IoU>={args.cache_iou_threshold:.2f} OR "
            f"adaptive center-distance match; "
            f"TTL={args.cache_ttl_sec:.1f}s; "
            f"refresh={args.cache_refresh_sec:.1f}s"
        )
        self.get_logger().info(f"Run ID: {self.run_id}")
        self.get_logger().info(f"Session results: {self.output_dir}")
        self.get_logger().info(
            "Saved frame name: frame_<processed_index>_yolo_<sec>_<nanosec>.jpg"
        )

    def close(self):
        self.stop_worker_event.set()
        self.latest_yolo_event.set()

        worker = getattr(self, "yolo_worker", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
            if worker.is_alive():
                self.get_logger().warning("Florence worker did not stop within 2 seconds.")

        try:
            self.result_file.close()
        except Exception:
            pass
        self.instance_lock.close()

    def write_csv_frame(self, frame_record: dict) -> None:
        """Append one CSV row per detected object; frame-level timestamps are repeated."""
        base = {
            "camera_timestamp_sec": frame_record.get("camera_timestamp_sec"),
            "yolo_timestamp_sec": frame_record.get("stamp_sec"),
            "final_timestamp": frame_record.get("result_wall_sec"),
            "num_skipped_YOLO": frame_record.get("latest_only_skipped_before"),
            "process_time_ms_afterYOLO": frame_record.get("result_age_from_yolo_ms"),
        }

        objects = frame_record.get("objects", [])

        # Keep one frame-level row even if there are no detected objects.
        if not objects:
            row = dict(base)
            row.update({
                "yolox_class": "",
                "existence_probability": "",
                "yolo_bbox": "",
                "florence_caption": "",
            })
            self.csv_writer.writerow(row)
            return

        for obj in objects:
            bbox = obj.get("yolox_bbox_px")
            if bbox is None:
                bbox_text = ""
            else:
                x1, y1, x2, y2 = bbox
                # Requested order inside a single CSV cell: x1, x2, y1, y2
                bbox_text = f"[{x1}, {x2}, {y1}, {y2}]"

            row = dict(base)
            row.update({
                "yolox_class": obj.get("yolox_class"),
                "existence_probability": obj.get("existence_probability"),
                "yolo_bbox": bbox_text,
                "florence_caption": obj.get("florence_caption") or "",
            })
            self.csv_writer.writerow(row)

    def image_callback(self, msg: RosImage) -> None:
        try:
            image = ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")
            return

        item = {
            "stamp_ns": stamp_ns(msg.header.stamp),
            "stamp_sec": stamp_sec(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "image": image,
        }
        with self.image_lock:
            self.image_buffer.append(item)

    def closest_image(self, target_stamp_ns: int):
        with self.image_lock:
            if not self.image_buffer:
                return None, None
            candidates = list(self.image_buffer)

        selected = min(
            candidates,
            key=lambda item: abs(item["stamp_ns"] - target_stamp_ns),
        )
        dt_sec = abs(selected["stamp_ns"] - target_stamp_ns) * 1e-9

        if dt_sec > self.args.max_image_time_diff:
            return None, dt_sec
        return selected, dt_sec

    @staticmethod
    def get_label(feature_object):
        classifications = feature_object.object.classification
        if not classifications:
            return 0, "UNKNOWN", None

        best = max(classifications, key=lambda item: float(item.probability))
        label = int(best.label)
        class_probability = float(best.probability)
        return label, LABEL_NAMES.get(label, f"LABEL_{label}"), class_probability

    @staticmethod
    def get_yolox_bbox(feature_object):
        roi = feature_object.feature.roi
        x1 = int(roi.x_offset)
        y1 = int(roi.y_offset)
        x2 = x1 + int(roi.width)
        y2 = y1 + int(roi.height)
        return x1, y1, x2, y2

    def purge_semantic_cache(self, now_ns: int) -> None:
        """Remove objects that have not been seen recently."""
        ttl_ns = int(self.args.cache_ttl_sec * 1e9)
        self.semantic_cache = [
            entry
            for entry in self.semantic_cache
            if now_ns - entry["last_seen_ns"] <= ttl_ns
        ]

    @staticmethod
    def cache_class_compatible(current_label: int, cached_label: int) -> bool:
        # UNKNOWN acts as a wildcard because low-confidence YOLO labels can
        # fluctuate even when the physical object is the same.
        return (
            current_label == cached_label
            or current_label == 0
            or cached_label == 0
        )

    def find_cache_match(
        self,
        label: int,
        bbox,
        now_ns: int,
        used_cache_ids: set,
    ):
        """
        Find the most plausible previously described object.

        We use both bbox IoU and bbox-center distance. IoU alone is unreliable
        for tiny distant objects, so center distance is used as a fallback.
        """
        self.purge_semantic_cache(now_ns)

        current_w = max(1, bbox[2] - bbox[0])
        current_h = max(1, bbox[3] - bbox[1])

        best = None
        best_score = -1.0

        for entry in self.semantic_cache:
            if entry["cache_id"] in used_cache_ids:
                continue

            if not self.cache_class_compatible(label, entry["label"]):
                continue

            cached_bbox = entry["bbox"]
            iou = bbox_iou(bbox, cached_bbox)
            center_distance = bbox_center_distance(bbox, cached_bbox)
            area_ratio = bbox_area_ratio(bbox, cached_bbox)

            cached_w = max(1, cached_bbox[2] - cached_bbox[0])
            cached_h = max(1, cached_bbox[3] - cached_bbox[1])

            adaptive_distance_limit = max(
                self.args.cache_center_distance_px,
                self.args.cache_center_scale
                * max(current_w, current_h, cached_w, cached_h),
            )

            spatial_match = (
                iou >= self.args.cache_iou_threshold
                or center_distance <= adaptive_distance_limit
            )
            size_match = area_ratio <= self.args.cache_max_area_ratio

            if not (spatial_match and size_match):
                continue

            distance_score = max(
                0.0,
                1.0 - center_distance / max(1.0, adaptive_distance_limit),
            )
            score = iou + 0.25 * distance_score

            if score > best_score:
                best_score = score
                best = {
                    "entry": entry,
                    "iou": iou,
                    "center_distance_px": center_distance,
                    "distance_limit_px": adaptive_distance_limit,
                    "area_ratio": area_ratio,
                }

        return best

    def update_cache_track(
        self,
        entry,
        label: int,
        bbox,
        now_ns: int,
    ) -> None:
        """Move an existing semantic cache track to the current YOLO bbox."""
        if label != 0:
            entry["label"] = label
        entry["bbox"] = tuple(bbox)
        entry["last_seen_ns"] = now_ns
        entry["last_seen_frame"] = self.frame_index

    def create_cache_entry(
        self,
        label: int,
        bbox,
        caption: str,
        now_ns: int,
    ):
        entry = {
            "cache_id": self.next_cache_id,
            "label": label,
            "bbox": tuple(bbox),
            "caption": caption,
            "caption_created_ns": now_ns,
            "last_seen_ns": now_ns,
            "last_seen_frame": self.frame_index,
        }
        self.next_cache_id += 1
        self.semantic_cache.append(entry)
        return entry

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
        input_ids, pixel_values, preprocess_ms, h2d_ms = self.prepare_florence_inputs(
            crop_bgr
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

    def yolox_callback(self, msg: DetectedObjectsWithFeature) -> None:
        """Store only the most recently received YOLO result."""
        receive_wall_ns = time.time_ns()

        with self.latest_yolo_lock:
            self.latest_yolo_seq += 1
            self.latest_yolo_msg = msg
            self.latest_yolo_receive_wall_ns = receive_wall_ns

        # Wake the worker. If it is already running Florence, this event simply
        # ensures it checks the newest message as soon as that work finishes.
        self.latest_yolo_event.set()

    def latest_yolo_worker(self) -> None:
        """Process the latest YOLO result only; never drain a stale backlog."""
        while not self.stop_worker_event.is_set():
            self.latest_yolo_event.wait(timeout=0.1)

            if self.stop_worker_event.is_set():
                break

            with self.latest_yolo_lock:
                msg = self.latest_yolo_msg
                seq = self.latest_yolo_seq
                receive_wall_ns = self.latest_yolo_receive_wall_ns

                # Consume the current notification. If a newer result arrives
                # after this lock is released, the callback sets the event again.
                self.latest_yolo_event.clear()

            if msg is None or seq <= self.last_processed_yolo_seq:
                continue

            skipped_before = max(
                0,
                seq - self.last_processed_yolo_seq - 1,
            )

            processing_start_wall_ns = time.time_ns()

            try:
                self.process_yolo_message(
                    msg=msg,
                    latest_only_seq=seq,
                    skipped_before=skipped_before,
                    receive_wall_ns=receive_wall_ns,
                    processing_start_wall_ns=processing_start_wall_ns,
                )
            except Exception as exc:
                self.get_logger().error(
                    f"Latest-only YOLO worker error: {exc}"
                )
            finally:
                self.last_processed_yolo_seq = seq

            # If new YOLO results arrived while this message was being
            # processed, wake immediately and jump to the newest one.
            with self.latest_yolo_lock:
                if self.latest_yolo_seq > self.last_processed_yolo_seq:
                    self.latest_yolo_event.set()

    def process_yolo_message(
        self,
        msg: DetectedObjectsWithFeature,
        latest_only_seq: int,
        skipped_before: int,
        receive_wall_ns: int,
        processing_start_wall_ns: int,
    ) -> None:
        callback_start = time.perf_counter()
        yolox_stamp_ns = stamp_ns(msg.header.stamp)
        selected_image, image_dt_sec = self.closest_image(yolox_stamp_ns)

        frame_record = {
            "frame_index": self.frame_index,
            "yolo_sec": int(msg.header.stamp.sec),
            "yolo_nanosec": int(msg.header.stamp.nanosec),
            "stamp_sec": stamp_sec(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "camera_timestamp_sec": (
                None if selected_image is None else selected_image["stamp_sec"]
            ),
            "camera_time_difference_ms": (
                None if image_dt_sec is None else image_dt_sec * 1000.0
            ),
            "latest_only_seq": latest_only_seq,
            "latest_only_skipped_before": skipped_before,
            "yolo_receive_wall_sec": receive_wall_ns * 1e-9,
            "processing_start_wall_sec": processing_start_wall_ns * 1e-9,
            "queue_wait_after_receive_ms": (
                processing_start_wall_ns - receive_wall_ns
            ) * 1e-6,
            "result_wall_sec": None,
            "result_age_from_yolo_ms": None,
            "objects": [],
            "frame_path": None,
            "frame_save_latency_ms": None,
        }

        print()
        print("=" * 110)
        print(
            f"[FRAME {self.frame_index:06d}] "
            f"YOLO timestamp={int(msg.header.stamp.sec)}."
            f"{int(msg.header.stamp.nanosec):09d} | "
            f"YOLOX objects={len(msg.feature_objects)} | "
            f"latest_seq={latest_only_seq} skipped={skipped_before} | "
            f"camera dt="
            f"{'N/A' if image_dt_sec is None else f'{image_dt_sec * 1000.0:.3f} ms'}"
        )

        if selected_image is None:
            print(
                f"[FRAME {self.frame_index:06d}] SKIP: "
                f"No camera image within {self.args.max_image_time_diff * 1000.0:.1f} ms."
            )
            frame_record["pipeline_latency_ms"] = (
                time.perf_counter() - callback_start
            ) * 1000.0

            result_wall_ns = time.time_ns()
            frame_record["result_wall_sec"] = result_wall_ns * 1e-9
            frame_record["result_age_from_yolo_ms"] = (
                result_wall_ns - yolox_stamp_ns
            ) * 1e-6

            self.write_csv_frame(frame_record)
            self.frame_index += 1
            return

        image_bgr = selected_image["image"]
        image_height, image_width = image_bgr.shape[:2]
        annotated = image_bgr.copy()
        used_cache_ids = set()

        for object_index, feature_object in enumerate(msg.feature_objects):
            confidence = float(feature_object.object.existence_probability)
            label, class_name, class_probability = self.get_label(feature_object)

            raw_bbox = clip_bbox(
                self.get_yolox_bbox(feature_object), image_width, image_height
            )
            raw_w = raw_bbox[2] - raw_bbox[0]
            raw_h = raw_bbox[3] - raw_bbox[1]

            uncertain = (
                label == 0 or confidence < self.args.confidence_threshold
            )

            record = {
                "object_index": object_index,
                "yolox_label": label,
                "yolox_class": class_name,
                "existence_probability": confidence,
                "classification_probability": class_probability,
                "yolox_bbox_px": list(raw_bbox),
                "yolox_bbox_size_px": {"width": raw_w, "height": raw_h},
                "florence_required": uncertain,
                "florence_triggered": False,
                "florence_crop_bbox_px": None,
                "florence_caption": None,
                "florence_latency_ms": None,
                "florence_timing": None,
                "semantic_source": "YOLOX",
                "cache_hit": False,
                "cache_id": None,
                "cache_caption_age_ms": None,
                "cache_match_iou": None,
                "cache_center_distance_px": None,
                "cache_distance_limit_px": None,
                "cache_area_ratio": None,
                "crop_path": None,
                "error": None,
            }

            if raw_w <= 1 or raw_h <= 1:
                record["error"] = "Invalid/empty YOLOX ROI."
                frame_record["objects"].append(record)
                continue

            cache_match = self.find_cache_match(
                label=label,
                bbox=raw_bbox,
                now_ns=yolox_stamp_ns,
                used_cache_ids=used_cache_ids,
            )

            # If this object was described earlier, keep its cached position
            # aligned even on frames where YOLO confidence is currently high.
            if cache_match is not None:
                matched_entry = cache_match["entry"]
                self.update_cache_track(
                    matched_entry,
                    label=label,
                    bbox=raw_bbox,
                    now_ns=yolox_stamp_ns,
                )
                used_cache_ids.add(matched_entry["cache_id"])

            if not uncertain:
                self.draw_bbox(
                    annotated,
                    raw_bbox,
                    f"{class_name} {confidence:.2f}",
                    (0, 220, 0),
                    thickness=2,
                )
                if cache_match is not None:
                    record["cache_id"] = cache_match["entry"]["cache_id"]

                print(
                    f"  [{object_index:02d}] YOLOX "
                    f"class={class_name:<12} conf={confidence:.3f} "
                    f"bbox={raw_bbox} size={raw_w}x{raw_h}"
                )
                frame_record["objects"].append(record)
                continue

            crop_bbox = expand_bbox_with_min_size(
                raw_bbox,
                image_width=image_width,
                image_height=image_height,
                expand_ratio=self.args.crop_expand_ratio,
                min_width=self.args.min_crop_width,
                min_height=self.args.min_crop_height,
            )
            record["florence_crop_bbox_px"] = list(crop_bbox)

            self.draw_bbox(
                annotated,
                raw_bbox,
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

            # Reuse the old caption if this is the same recently seen object.
            if cache_match is not None:
                cache_entry = cache_match["entry"]
                caption_age_ms = (
                    yolox_stamp_ns - cache_entry["caption_created_ns"]
                ) * 1e-6

                record["cache_id"] = cache_entry["cache_id"]
                record["cache_caption_age_ms"] = caption_age_ms
                record["cache_match_iou"] = cache_match["iou"]
                record["cache_center_distance_px"] = (
                    cache_match["center_distance_px"]
                )
                record["cache_distance_limit_px"] = (
                    cache_match["distance_limit_px"]
                )
                record["cache_area_ratio"] = cache_match["area_ratio"]

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
                        f"YOLOX_class={class_name:<12} "
                        f"conf={confidence:.3f} | "
                        f"IoU={cache_match['iou']:.3f} "
                        f"center={cache_match['center_distance_px']:.1f}px/"
                        f"{cache_match['distance_limit_px']:.1f}px "
                        f"area_ratio={cache_match['area_ratio']:.2f} | "
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
                record["florence_triggered"] = True
                record["semantic_source"] = "Florence"
                record["florence_caption"] = caption
                record["florence_latency_ms"] = timings["generate_ms"]
                record["florence_timing"] = timings

                if cache_match is not None:
                    cache_entry = cache_match["entry"]
                    cache_entry["caption"] = caption
                    cache_entry["caption_created_ns"] = yolox_stamp_ns
                    self.update_cache_track(
                        cache_entry,
                        label=label,
                        bbox=raw_bbox,
                        now_ns=yolox_stamp_ns,
                    )
                else:
                    cache_entry = self.create_cache_entry(
                        label=label,
                        bbox=raw_bbox,
                        caption=caption,
                        now_ns=yolox_stamp_ns,
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

                print(
                    f"  [{object_index:02d}] FLORENCE "
                    f"YOLOX_class={class_name:<12} conf={confidence:.3f} "
                    f"bbox={raw_bbox} crop={crop_bbox} | "
                    f"caption={caption!r} | "
                    f"pre={timings['preprocess_ms']:.1f} ms "
                    f"H2D={timings['h2d_ms']:.1f} ms "
                    f"generate={timings['generate_ms']:.1f} ms "
                    f"decode={timings['decode_ms']:.1f} ms "
                    f"total={timings['total_ms']:.1f} ms"
                )

                if self.args.save_crops:
                    crop_path = self.crop_dir / (
                        f"frame_{self.frame_index:06d}_"
                        f"yolo_{int(msg.header.stamp.sec)}_"
                        f"{int(msg.header.stamp.nanosec):09d}_"
                        f"object_{object_index:02d}_"
                        f"{class_name.lower()}_{confidence:.3f}.jpg"
                    )
                    cv2.imwrite(
                        str(crop_path),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
                    )
                    record["crop_path"] = str(crop_path)

            except Exception as exc:
                record["error"] = str(exc)
                print(f"  [{object_index:02d}] FLORENCE ERROR: {exc}")

            frame_record["objects"].append(record)

        save_start = time.perf_counter()
        cv2.putText(
            annotated,
            f"YOLOX frame {self.frame_index:06d} | camera_dt={image_dt_sec * 1000.0:.1f} ms",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        yolo_sec = int(msg.header.stamp.sec)
        yolo_nanosec = int(msg.header.stamp.nanosec)
        frame_path = self.frame_dir / (
            f"frame_{self.frame_index:06d}_"
            f"yolo_{yolo_sec}_{yolo_nanosec:09d}.jpg"
        )
        cv2.imwrite(
            str(frame_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )

        frame_record["frame_path"] = str(frame_path)
        frame_record["frame_save_latency_ms"] = (
            time.perf_counter() - save_start
        ) * 1000.0
        frame_record["pipeline_latency_ms"] = (
            time.perf_counter() - callback_start
        ) * 1000.0

        result_wall_ns = time.time_ns()
        frame_record["result_wall_sec"] = result_wall_ns * 1e-9
        frame_record["result_age_from_yolo_ms"] = (
            result_wall_ns - yolox_stamp_ns
        ) * 1e-6

        self.write_csv_frame(frame_record)

        print(
            f"[FRAME {self.frame_index:06d}] "
            f"pipeline={frame_record['pipeline_latency_ms']:.1f} ms | "
            f"frame save={frame_record['frame_save_latency_ms']:.1f} ms | "
            f"result age={frame_record['result_age_from_yolo_ms']:.1f} ms | "
            f"semantic cache={len(self.semantic_cache)}"
        )
        self.frame_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use 2D detector class/confidence/ROI directly. "
            "Run Florence only for UNKNOWN or low-confidence objects."
        )
    )

    parser.add_argument("--image_topic", default="/rsu1/gate/image_raw")
    parser.add_argument("--yolox_topic", default="/rsu1/gate/yolox/objects")
    parser.add_argument(
        "--detector_topic",
        default=None,
        help="Generic alias for --yolox_topic for any message-compatible detector.",
    )
    parser.add_argument(
        "--florence_model", default="florence-community/Florence-2-base-ft"
    )
    parser.add_argument(
        "--florence_prompt",
        default="Describe what is shown in the image with heading direction.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.4)

    parser.add_argument(
        "--cache_iou_threshold",
        type=float,
        default=0.20,
        help="Reuse cache when bbox IoU is at least this value.",
    )
    parser.add_argument(
        "--cache_center_distance_px",
        type=float,
        default=20.0,
        help="Minimum allowed bbox-center displacement for cache matching.",
    )
    parser.add_argument(
        "--cache_center_scale",
        type=float,
        default=1.5,
        help="Adaptive center-distance limit relative to bbox size.",
    )
    parser.add_argument(
        "--cache_max_area_ratio",
        type=float,
        default=4.0,
        help="Reject matches whose bbox areas differ by more than this ratio.",
    )
    parser.add_argument(
        "--cache_ttl_sec",
        type=float,
        default=2.0,
        help="Forget an object if it has not been matched for this many seconds.",
    )
    parser.add_argument(
        "--cache_refresh_sec",
        type=float,
        default=5.0,
        help="Force a fresh Florence inference after this caption age.",
    )
    parser.add_argument(
        "--crop_expand_ratio",
        type=float,
        default=0.5,
        help="0.5 means 50 percent larger than the YOLO bbox before min-size enforcement.",
    )
    parser.add_argument("--min_crop_width", type=int, default=64)
    parser.add_argument("--min_crop_height", type=int, default=64)
    parser.add_argument(
        "--max_image_time_diff",
        type=float,
        default=0.10,
        help="Maximum YOLOX-result vs raw-camera timestamp difference in seconds.",
    )
    parser.add_argument("--image_buffer_size", type=int, default=120)
    parser.add_argument("--florence_max_new_tokens", type=int, default=15)
    parser.add_argument("--save_crops", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument(
        "--output_dir",
        default="/autoware_runtime/hyelin/results",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Unique result-session name; defaults to a JST timestamp plus PID.",
    )
    args = parser.parse_args()
    if args.detector_topic:
        args.yolox_topic = args.detector_topic
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = YoloxFlorenceNode(args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()