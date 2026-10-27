# new-dt-server live PointPainting calibration snapshot

Captured from the running `aw` and `yoloxcar` containers on 2026-09-13.

- LiDAR transform source: running `nd_start_ef.sh` / static TF process.
- Camera extrinsic source: running static TF processes used by PointPainting.
- Camera intrinsic source: live `/cam/taki_gate/camera_info` and
  `/cam/taki_intersection/camera_info`, cross-checked against
  `/yolox_sidecar.py` in the running camera containers.
- Remote frames `taki_gate_camera` and `taki_intersection_camera` map to the
  local frames `gate_camera` and `intersection_camera`.

The RSU1/Taki LiDAR transform already matched the local runtime exactly.
The canonical local camera YAML files were updated from this snapshot.
