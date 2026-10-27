"""Isolated-domain integration test of strict pairing and latest-cloud rate control."""
import os
import struct
import subprocess
import time

os.environ['ROS_DOMAIN_ID'] = '130'
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


def main():
    rclpy.init()
    node = Node('fusion_latest_regression')
    received, limited = [], []
    subscriptions = [
        node.create_subscription(PointCloud2, '/sensing/lidar/concatenated/pointcloud',
                                 lambda msg: received.append(msg), qos_profile_sensor_data),
        node.create_subscription(PointCloud2, '/sensing/lidar/centerpoint/pointcloud',
                                 lambda msg: limited.append(msg), qos_profile_sensor_data)]
    publishers = [node.create_publisher(PointCloud2, f'/rsu{n}/sensing/lidar/pointcloud',
                                       qos_profile_sensor_data) for n in (1, 4)]
    process = subprocess.Popen([
        '/autoware_runtime/install/transport_upgrade_20261017/lidar_fusion_adapter/lib/lidar_fusion_adapter/strict_front_fusion_node',
        '--ros-args', '-p', 'centerpoint_max_hz:=15.0'], stdout=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not all(pub.get_subscription_count() for pub in publishers):
            assert process.poll() is None and time.monotonic() < deadline, 'Discovery failed'
            rclpy.spin_once(node, timeout_sec=.05)
        cloud = PointCloud2()
        cloud.header.frame_id = 'lidar_fusion'
        cloud.width = cloud.height = 1
        cloud.point_step = cloud.row_step = 16
        cloud.fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                        for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
        cloud.data = struct.pack('ffff', 1, 2, 3, 4)
        end = time.monotonic() + 5
        next_frame = time.monotonic()
        while time.monotonic() < end:
            if time.monotonic() >= next_frame:
                cloud.header.stamp = node.get_clock().now().to_msg()
                for publisher in publishers:
                    publisher.publish(cloud)
                next_frame += .05
            rclpy.spin_once(node, timeout_sec=.002)
        def spin(seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
        spin(.2)
        assert len(received) >= 85, f'Fusion lost synthetic pairs: {len(received)}'
        assert 60 <= len(limited) <= 79, f'Unexpected limited output count: {len(limited)}'
        stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in received}
        limited_stamps = [(m.header.stamp.sec, m.header.stamp.nanosec) for m in limited]
        assert len(set(limited_stamps)) == len(limited_stamps), 'Duplicate limited frame'
        assert all(stamp in stamps for stamp in limited_stamps), 'Re-stamped or invented frame'
        before = len(limited)
        spin(.3)
        assert len(limited) == before, 'Stale cloud republished after input stopped'
        assert all(m.width == 2 and m.point_step == 16 for m in received)
        print(f'PASS: {len(received)} strict pairs, {len(limited)} latest inputs; stamps retained; no stale repeats.')
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
