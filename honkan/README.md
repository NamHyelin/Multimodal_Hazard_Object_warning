# Autoware runtime overlay

This host-mounted ROS 2 overlay keeps sensor integration and the TensorRT 10.16
CenterPoint and YOLOX builds separate from the immutable Autoware installation at
`/opt/autoware`.

## Runtime layout

- `src/`: RoboSense driver, strict LiDAR fusion, and copied CenterPoint/YOLOX sources
- `build/`, `install/`, `log/`: persistent colcon artifacts
- `models/lidar_centerpoint/`: official CenterPoint ONNX and Orin/TRT 10.16 engines
- `models/tensorrt_yolox/`: official YOLOX-Tiny ONNX, labels, and Orin/TRT 10.16 engine
- `vendor/tensorrt-10.16/include/`: headers used only by the devel build
- `map/`, `calibration/`, `config/`: map, TF, and runtime parameters
- `scripts/`: build, start, stop, enter, and status commands

## Normal operation

The local pipeline uses ROS domain `30` by default for both containers. This
keeps unrelated ROS 2 nodes on the shared KLab network out of the local graph.
Set `ROS_DOMAIN_ID=30` in host-side ROS CLI sessions that need to inspect or
subscribe to this pipeline.

```bash
cd /home/honkanmae/SSD/workspace/tian_ws/autoware_runtime
./scripts/start/start.sh
./scripts/start/status.sh
./scripts/enter_autoware_runtime_bash/enter.sh
```

`start.sh` starts the runtime and RoboSense containers, the dual-camera
receiver, strict dual-LiDAR fusion, CenterPoint Tiny, Multi Object Tracker, and
dual-camera YOLOX in that order. Topics:

```text
/sensing/lidar/concatenated/pointcloud
  -> /perception/object_recognition/detection/centerpoint/objects
  -> /perception/object_recognition/tracking/objects
```

CenterPoint is demand-driven: it runs inference only while this output has a
subscriber. The tracker provides that persistent subscription.

YOLOX-Tiny subscribes directly to the two raw Taki camera streams and publishes
independent 2D detections plus full-frame images with detection boxes:

```text
/rsu1/gate/image_raw
  -> /rsu1/gate/yolox/objects
  -> /tensorrt_yolox_gate/out/image
/rsu1/intersection/image_raw
  -> /rsu1/intersection/yolox/objects
  -> /tensorrt_yolox_intersection/out/image
```

The compact YOLOX topic profile is controlled in
`src/autoware_tensorrt_yolox/config/yolox_tiny.param.yaml`:

```yaml
publish_visualization: true
publish_masks: false
publish_debug_metrics: false
use_image_transport: false
```

Set `publish_masks` to restore mask topics, `publish_debug_metrics` to restore
timing topics, or `use_image_transport` to restore compressed/theora transport
topics. Restart YOLOX after changing these startup parameters.

Manage perception components individually with:

```bash
./scripts/stop/stop_centerpoint.sh
./scripts/start/start_centerpoint.sh
./scripts/stop/stop_tracker.sh
./scripts/start/start_tracker.sh
./scripts/stop/stop_yolox.sh
./scripts/start/start_yolox.sh
./scripts/yolox_debug/capture_yolox_frame.sh
tail -f logs/centerpoint.log
tail -f logs/tracker.log
tail -f logs/yolox.log
```

`capture_yolox_frame.sh` waits for one annotated frame from each YOLOX camera
and overwrites `captures/yolox/gate_yolox.png` and
`captures/yolox/intersection_yolox.png`. Use `--timeout SECONDS` to change the
default 10-second timeout.

The tracker uses `ego_source=tf`, world frame `map`, ego frame `base_link`, and
the official `lidar_centerpoint` input-channel profile.

## Building

Build normal runtime packages inside `tian_autoware_runtime`:

```bash
docker exec tian_autoware_runtime bash -lc /autoware_runtime/scripts/build/build.sh
```

Build the TensorRT 10.16 CenterPoint overlay from the host through the existing
devel container:

```bash
./scripts/build/build_centerpoint.sh
./scripts/build/build_yolox.sh
```

Do not build `autoware_tensorrt_common`, `autoware_lidar_centerpoint`, or
`autoware_tensorrt_yolox` with the runtime container's TensorRT 10.3 headers.
The deployed binaries, runtime libraries, and cached engines must all use
TensorRT 10.16.2.
