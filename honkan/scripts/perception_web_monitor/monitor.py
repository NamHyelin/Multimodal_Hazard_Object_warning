#!/usr/bin/env python3
"""Read-only ROS perception viewer; bounded queues and shared JPEG encoding."""
import collections
import functools
import fcntl
import json
import socket
import threading
import time
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from autoware_perception_msgs.msg import DetectedObjects
from tier4_perception_msgs.msg import DetectedObjectsWithFeature
from tf2_ros import Buffer, TransformListener
from jpeg_transport import read_frame
from projection import box_geometry

ROOT = Path('/autoware_runtime')
CFG = json.loads((ROOT / 'config/perception_web_monitor/monitor.json').read_text())
LABELS = ['UNKNOWN', 'CAR', 'TRUCK', 'BUS', 'TRAILER', 'MOTORCYCLE', 'BICYCLE', 'PEDESTRIAN']
EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]

def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

def rotation(q):
    v = np.array([q.x, q.y, q.z, q.w], dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = v / n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def nearest(buf, t):
    if not buf:
        return None, None
    msg = min(buf, key=lambda m: abs(stamp(m)-t))
    dt = stamp(msg)-t
    return (msg if abs(dt) <= CFG['sync_tolerance_sec'] else None), dt

def label(obj):
    c = max(obj.classification, key=lambda x: x.probability, default=None)
    return LABELS[c.label] if c and c.label < len(LABELS) else 'UNKNOWN'

class Monitor(Node):
    def __init__(self):
        super().__init__('perception_web_monitor')
        self.lock = threading.RLock()
        self.lidar_box_mode = CFG.get('lidar_box_mode', '2d')
        if self.lidar_box_mode not in ('2d', '3d'):
            raise ValueError('lidar_box_mode must be 2d or 3d')
        self.stop_event = threading.Event()
        self.radar = collections.deque(maxlen=40)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.cams = {}
        self.associations = {name: collections.deque(maxlen=40) for name in ('gate', 'intersection')}
        self.association_seen = 0
        self.create_subscription(String, '/perception/id_association/results', self.association_cb, 10)
        self.create_subscription(DetectedObjects, CFG['centerpoint_topic'], self.radar_cb, qos_profile_sensor_data)
        self.workers = []
        self.image_transport = CFG.get('image_transport', 'ros_raw')
        if self.image_transport not in ('jpeg_ipc', 'ros_raw'):
            raise ValueError('Unknown image transport: ' + self.image_transport)
        for name in ('gate', 'intersection'):
            self.cams[name] = dict(images=collections.deque(maxlen=3), yolo=collections.deque(maxlen=20), info=None,
                                   jpeg=None, seq=0, last_output=0, received=0, times=collections.deque(maxlen=50),
                                   input_times=collections.deque(maxlen=120), skipped=0, reader_error=None, status={})
            if self.image_transport == 'ros_raw':
                self.create_subscription(Image, f'/rsu1/{name}/image_raw', functools.partial(self.image_cb, name), qos_profile_sensor_data)
            else:
                reader = threading.Thread(target=self.jpeg_reader, args=(name,), daemon=True)
                reader.start()
                self.workers.append(reader)
            self.create_subscription(CameraInfo, f'/rsu1/{name}/camera_info', functools.partial(self.info_cb, name), qos_profile_sensor_data)
            self.create_subscription(DetectedObjectsWithFeature, f'/rsu1/{name}/yolo26s/objects', functools.partial(self.yolo_cb, name), qos_profile_sensor_data)
            th = threading.Thread(target=self.worker, args=(name,), daemon=True)
            th.start()
            self.workers.append(th)

    def radar_cb(self, msg):
        with self.lock:
            self.radar.append(msg)

    def association_cb(self, msg):
        try:
            result = json.loads(msg.data)
            with self.lock:
                self.associations[result['camera']].append(result)
                self.association_seen = time.monotonic()
        except (ValueError, KeyError, TypeError):
            pass

    def image_cb(self, name, msg):
        self.enqueue_image(name, msg, None)

    def enqueue_image(self, name, msg, jpeg):
        with self.lock:
            cam = self.cams[name]
            at = time.monotonic()
            if len(cam['images']) == cam['images'].maxlen:
                cam['skipped'] += 1
            cam['images'].append((at, msg, jpeg))
            cam['received'] += 1
            cam['input_times'].append(at)
            cam['reader_error'] = None

    def jpeg_reader(self, name):
        while not self.stop_event.is_set():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as source:
                    source.settimeout(1.0)
                    source.connect('\0perception_' + name + '_jpeg_v1')
                    while not self.stop_event.is_set():
                        sec, nsec, jpeg = read_frame(source)
                        # Preserve the ROS Image stamp, rather than re-stamping at arrival.
                        msg = SimpleNamespace(header=SimpleNamespace(
                            stamp=SimpleNamespace(sec=sec, nanosec=nsec), frame_id=name + '_camera'))
                        self.enqueue_image(name, msg, jpeg)
            except (OSError, EOFError, ValueError) as exc:
                with self.lock:
                    self.cams[name]['reader_error'] = str(exc)
                self.stop_event.wait(.5)

    def info_cb(self, name, msg):
        with self.lock:
            self.cams[name]['info'] = msg

    def yolo_cb(self, name, msg):
        with self.lock:
            self.cams[name]['yolo'].append(msg)

    def worker(self, name):
        cam = self.cams[name]
        next_output = 0
        while not self.stop_event.wait(0.005):
            now = time.monotonic()
            if now < next_output:
                continue
            with self.lock:
                # Extra wait is only active while association metadata is arriving.
                wait = max(CFG['sync_wait_sec'], .14) if now-self.association_seen < 2 else CFG['sync_wait_sec']
                ready = [(at, m, jpg) for at, m, jpg in cam['images'] if now-at >= wait]
                if not ready:
                    continue
                at, msg, jpeg = ready[-1]
                consumed = 0
                while cam['images'] and cam['images'][0][0] <= at:
                    cam['images'].popleft()
                    consumed += 1
                cam['skipped'] += max(0, consumed - 1)
                yolo, yd = nearest(cam['yolo'], stamp(msg))
                radar, rd = nearest(self.radar, stamp(msg))
                info = cam['info']
            # Fixed cadence avoids repeatedly resetting the deadline against a
            # source that itself runs at 10 Hz and has small arrival jitter.
            next_output = max(next_output + 1/CFG['fps'], now)
            try:
                decoded = None
                if jpeg is not None:
                    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None:
                        raise ValueError('JPEG decode failed')
                    msg.height, msg.width = decoded.shape[:2]
                    msg.encoding = 'bgr8'
                frame, state = self.render(msg, info, yolo, radar, decoded)
                state.update(yolo_dt_ms=None if yd is None else round(yd*1000, 1),
                             lidar_dt_ms=None if rd is None else round(rd*1000, 1),
                             image_stamp=stamp(msg), image_age_ms=round((self.get_clock().now().nanoseconds*1e-9-stamp(msg))*1000, 1))
                text = f'{name}  YOLO: {state["yolo"]}  LiDAR: {state["lidar"]} {state["lidar_box_mode"]}'
                if 'association' in state:
                    text += f'  ID: {state["association"]} {state["confirmed"]}/{state["associated"]}'
                cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
                width = min(CFG['width'], frame.shape[1])
                frame = cv2.resize(frame, (width, round(frame.shape[0]*width/frame.shape[1])))
                ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CFG['jpeg_quality']])
                if not ok:
                    raise ValueError('JPEG encode failed')
                with self.lock:
                    cam.update(jpeg=data.tobytes(), seq=cam['seq']+1, last_output=time.monotonic(), status=state)
                    cam['times'].append(time.monotonic())
            except Exception as exc:
                with self.lock:
                    cam['status'] = {'error': str(exc)}

    def render(self, msg, info, yolo, radar, decoded=None):
        if msg.encoding not in ('bgr8', 'rgb8'):
            raise ValueError('Unsupported image encoding '+msg.encoding)
        im = decoded if decoded is not None else np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width*3].reshape(msg.height,msg.width,3).copy()
        if msg.encoding == 'rgb8':
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        with self.lock:
            box_mode = self.lidar_box_mode
        state = {'yolo': 'UNSYNC' if yolo is None else 'OK', 'lidar': 'UNSYNC' if radar is None else 'OK', 'yolo_boxes': 0, 'lidar_boxes': 0,
                 'lidar_box_mode': box_mode}
        if yolo is not None and yolo.header.frame_id != msg.header.frame_id:
            state['yolo'] = 'FRAME_MISMATCH'
            yolo = None
        if yolo is not None:
            camera = msg.header.frame_id.removesuffix('_camera')
            with self.lock:
                candidates = list(self.associations.get(camera, []))
            # Exact source stamp and ROI index: never attach a previous frame's ID.
            assoc = next((a for a in reversed(candidates) if
                          a['stamp_sec'] == yolo.header.stamp.sec and
                          a['stamp_nanosec'] == yolo.header.stamp.nanosec and
                          yolo.header.stamp.sec == msg.header.stamp.sec and
                          yolo.header.stamp.nanosec == msg.header.stamp.nanosec), None)
            ids = {m['yolo_index']: m for m in assoc['matches']} if assoc else {}
            state.update(association='WAIT' if assoc is None else assoc['status'],
                         associated=len(ids), confirmed=sum(bool(m['confirmed']) for m in ids.values()))
            for index, item in enumerate(yolo.feature_objects):
                roi, obj = item.feature.roi, item.object
                if obj.existence_probability < CFG['yolo_score']:
                    continue
                a = (min(msg.width-1, int(roi.x_offset)), min(msg.height-1, int(roi.y_offset)))
                b = (min(msg.width-1, int(roi.x_offset+roi.width)), min(msg.height-1, int(roi.y_offset+roi.height)))
                cv2.rectangle(im, a, b, (0,220,0), 2)
                match = ids.get(index)
                suffix = '' if match is None else (' ID#' + match['display_id'] + ('' if match['confirmed'] else '?'))
                cv2.putText(im, f'Y {label(obj)} {obj.existence_probability:.2f}{suffix}', (a[0],max(45,a[1]-5)), 0, .5, (0,220,0), 1)
                if match is not None:
                    x1,y1,x2,y2 = [int(v) for v in match['projected_box']]
                    cv2.rectangle(im,(x1,y1),(x2,y2),(220,80,220),1)
                state['yolo_boxes'] += 1
        if radar is None:
            return im, state
        if info is None or info.width != msg.width or info.height != msg.height or info.header.frame_id != msg.header.frame_id:
            state['lidar'] = 'CALIBRATION_MISSING_OR_MISMATCH'
            return im, state
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial') or info.k[0] <= 0:
            state['lidar'] = 'INVALID_CALIBRATION'
            return im, state
        try:
            tf = self.tf.lookup_transform(msg.header.frame_id, radar.header.frame_id, Time.from_msg(radar.header.stamp)).transform
        except Exception:
            state['lidar'] = 'TF_UNAVAILABLE'
            return im, state
        rot = rotation(tf.rotation)
        trans = np.array([tf.translation.x,tf.translation.y,tf.translation.z])
        k = np.array(info.k).reshape(3,3)
        d = np.array(info.d)
        for obj in radar.objects:
            if obj.existence_probability < CFG['lidar_score']:
                continue
            pose = obj.kinematics.pose_with_covariance.pose
            dims = np.array([obj.shape.dimensions.x,obj.shape.dimensions.y,obj.shape.dimensions.z])
            if not np.isfinite(dims).all() or np.any(dims <= 0):
                continue
            corners = np.array([[1 if i & bit else -1 for bit in (1,2,4)] for i in range(8)])*dims/2
            r = rotation(pose.orientation) if obj.kinematics.orientation_availability else np.eye(3)
            corners = (corners @ r.T + [pose.position.x,pose.position.y,pose.position.z]) @ rot.T + trans
            segments, envelope = box_geometry(corners, k, d, msg.width, msg.height)
            if box_mode == '2d':
                visible = envelope is not None
                if visible:
                    cv2.rectangle(im, envelope[0], envelope[1], (255,255,0), 2)
            else:
                visible = bool(segments)
                for p,q in segments:
                    cv2.line(im,p,q,(255,255,0),2)
            state['lidar_boxes'] += int(visible)
        return im,state

    def display_settings(self, mode=None):
        with self.lock:
            if mode is not None:
                if mode not in ('2d', '3d'):
                    raise ValueError('lidar_box_mode must be 2d or 3d')
                self.lidar_box_mode = mode
            return {'lidar_box_mode': self.lidar_box_mode}

    def status(self):
        with self.lock:
            out = {}
            for name,c in self.cams.items():
                ts = c['times']
                age = time.monotonic()-c['last_output'] if c['last_output'] else None
                fps = (len(ts)-1)/(ts[-1]-ts[0]) if len(ts)>1 and age < 2 else 0
                its = c['input_times']
                input_fps = (len(its)-1)/(its[-1]-its[0]) if len(its)>1 and time.monotonic()-its[-1]<2 else 0
                gaps = [1000*(b-a) for a,b in zip(its, list(its)[1:])]
                out[name] = dict(c['status'], fps=round(fps,2), output_age_sec=None if age is None else round(age,2),
                                 received=c['received'], online=age is not None and age<2,
                                 image_transport=self.image_transport, input_fps=round(input_fps,2),
                                 input_gap_max_ms=round(max(gaps,default=0),1), skipped_frames=c['skipped'],
                                 reader_error=c['reader_error'])
            return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.path != '/api/display':
            self.send_error(404)
            return
        try:
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Expected application/json')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 256:
                raise ValueError('Invalid request length')
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict) or set(request) != {'lidar_box_mode'}:
                raise ValueError('Expected lidar_box_mode only')
            if request['lidar_box_mode'] not in ('2d', '3d'):
                raise ValueError('lidar_box_mode must be 2d or 3d')
            body = json.dumps(self.server.node.display_settings(request['lidar_box_mode'])).encode()
        except (ValueError, TypeError) as exc:
            self.send_error(400, str(exc))
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/api/status','/healthz','/api/display'):
            body = json.dumps(self.server.node.display_settings() if self.path == '/api/display' else self.server.node.status()).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
        elif self.path == '/':
            body = Path(__file__).with_name('web').joinpath('index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
        elif self.path in ('/stream/gate.mjpg','/stream/intersection.mjpg'):
            name = self.path.split('/')[-1].split('.')[0]
            self.connection.settimeout(3)
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            seq = -1
            try:
                while not self.server.node.stop_event.wait(.02):
                    with self.server.node.lock:
                        c = self.server.node.cams[name]
                        jpg, current = c['jpeg'], c['seq']
                    if jpg is None or current == seq:
                        continue
                    seq = current
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpg)).encode()+b'\r\n\r\n'+jpg+b'\r\n')
                    self.wfile.flush()
            except (OSError, TimeoutError):
                pass
            return
        else:
            self.send_error(404)
            return
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(body)

def main():
    instance = open('/tmp/perception_web_monitor.lock', 'a')
    fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cv2.setNumThreads(1)
    rclpy.init()
    node = Monitor()
    server = ThreadingHTTPServer((CFG['bind'],CFG['port']), Handler)
    server.daemon_threads = True
    server.node = node
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'Perception monitor listening on {CFG["bind"]}:{CFG["port"]}',flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.stop_event.set()
        server.shutdown()
        server.server_close()
        for th in node.workers:
            th.join(timeout=3)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
