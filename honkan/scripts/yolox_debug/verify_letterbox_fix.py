#!/usr/bin/env python3
"""Replay saved 1080p frames through old/fixed engines on isolated ROS topics.

Run inside the runtime container after sourcing the V2 environment. No live
camera topics are published. Both temporary inference processes are cleaned up.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tier4_perception_msgs.msg import DetectedObjectsWithFeature

ROOT = Path('/autoware_runtime')
MODELS = ROOT / 'models/perception_standard_v2/yolo26s'
OUTPUT = ROOT / 'logs/perception_standard_v2/letterbox_regression'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-model', type=Path,
                        default=MODELS / 'yolo26s_autoware.onnx')
    parser.add_argument('--fixed-model', type=Path,
                        default=MODELS / 'yolo26s_autoware.corrected.onnx')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Node('letterbox_coordinate_regression')
    received = {'old': {}, 'fixed': {}}
    processes, logs, subscriptions = [], [], []
    publisher = node.create_publisher(Image, '/letterbox_regression/image', 1)
    try:
        for variant, model_path in [('old', args.baseline_model),
                                    ('fixed', args.fixed_model)]:
            def callback(msg, variant=variant):
                received[variant][msg.header.stamp.sec] = msg
            subscriptions.append(node.create_subscription(
                DetectedObjectsWithFeature, f'/letterbox_regression/{variant}',
                callback, qos_profile_sensor_data))
            log = open(OUTPUT / f'{variant}.log', 'w')
            logs.append(log)
            processes.append(subprocess.Popen([
                str(Path(os.environ['PSV2_INSTALL']) / 'autoware_tensorrt_yolox/lib/'
                    'autoware_tensorrt_yolox/autoware_tensorrt_yolox_node_exe'),
                '--ros-args', '--params-file',
                str(ROOT / 'config/perception_standard_v2/yolo26s_standard.param.yaml'),
                '-p', f'model_path:={model_path}',
                '-p', f'label_path:={MODELS / "coco80_labels.txt"}',
                '-p', f'roi_remap_path:={MODELS / "coco80_to_autoware.csv"}',
                '-r', f'__node:=letterbox_regression_{variant}',
                '-r', '~/in/image:=/letterbox_regression/image',
                '-r', f'~/out/objects:=/letterbox_regression/{variant}',
                '-r', f'~/out/image:=/letterbox_regression/{variant}/image',
            ], stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 60
        while publisher.get_subscription_count() < 2:
            assert all(p.poll() is None for p in processes), 'Inference process exited'
            assert time.monotonic() < deadline, 'Inference subscription timeout'
            rclpy.spin_once(node, timeout_sec=.2)
        # Frames 70--73 contain a distant pedestrian confirmed by the saved
        # YOLO26 reference results; the beginning of this sequence is empty.
        paths = sorted((ROOT / 'yolo_benchmark/data/gate_20260908/images').glob('*.jpg'))[70:74]
        assert len(paths) == 4, 'Missing replay images'
        records, matches = [], []
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            assert image is not None and image.shape[:2] == (1080, 1920), path
            message = Image()
            message.header.stamp.sec = 1000 + index
            message.header.frame_id = 'gate_camera'
            message.height, message.width = image.shape[:2]
            message.encoding, message.step = 'bgr8', image.shape[1] * 3
            message.data = image.tobytes()
            key = message.header.stamp.sec
            deadline = time.monotonic() + 30
            next_publish = 0
            while not all(key in received[v] for v in received):
                assert time.monotonic() < deadline, 'Replay detection timeout'
                assert all(p.poll() is None for p in processes), 'Inference process exited'
                if time.monotonic() >= next_publish:
                    publisher.publish(message)
                    next_publish = time.monotonic() + 1
                rclpy.spin_once(node, timeout_sec=.1)
            boxes = {}
            for variant in received:
                boxes[variant] = []
                for item in received[variant][key].feature_objects:
                    if item.object.existence_probability < .25:
                        continue
                    roi = item.feature.roi
                    label = max(item.object.classification, key=lambda c: c.probability).label
                    boxes[variant].append((label, np.array([
                        roi.x_offset, roi.y_offset, roi.x_offset + roi.width,
                        roi.y_offset + roi.height], dtype=float)))
            unused = set(range(len(boxes['fixed'])))
            for label, old in boxes['old']:
                if old[1] <= 68 or old[3] >= 1078:
                    continue  # Border clamping is not a pure translation.
                candidates = [(np.max(np.abs(old - fixed - [0, 68, 0, 68])), j)
                              for j, (other_label, fixed) in enumerate(boxes['fixed'])
                              if j in unused and label == other_label]
                if not candidates:
                    continue
                error, j = min(candidates)
                if error > 4:
                    continue  # Allow small FP16 engine rebuild differences.
                unused.remove(j)
                delta = old - boxes['fixed'][j][1]
                matches.append(delta.tolist())
            for variant in boxes:
                canvas = image.copy()
                for _, box in boxes[variant]:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                assert cv2.imwrite(str(OUTPUT / f'{index}_{variant}.jpg'), canvas)
            records.append({'image': str(path), 'old_boxes': len(boxes['old']),
                            'fixed_boxes': len(boxes['fixed'])})
        assert len(matches) >= 3, f'Insufficient comparable detections: {len(matches)}'
        median = np.median(matches, axis=0)
        np.testing.assert_allclose(median, [0, 68, 0, 68], atol=2)
        result = {'frames': records, 'matched_boxes': len(matches),
                  'median_old_minus_fixed_xyxy': median.tolist(), 'status': 'PASS'}
        (OUTPUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
