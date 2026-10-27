# 本机双路感知网页监控

运行在 `tian_autoware_runtime` 中；需先启动 Perception Standard V2。

## 文件布局

```text
config/perception_web_monitor/monitor.json  # 地址、输出尺寸、帧率、同步阈值
config/perception_web_monitor/cyclonedds.xml # 监控专用 DDS 数据单播
scripts/perception_web_monitor/
  monitor.py                              # ROS 订阅、三维框投影与 HTTP 服务
  web/index.html                          # 双路监控页面
  start.sh / stop.sh / status.sh           # 在宿主机执行
  README.md
logs/perception_web_monitor/monitor.log
```

## 使用

```bash
cd /mnt/SSD/workspace/tian_ws/autoware_runtime
bash scripts/perception_web_monitor/start.sh
bash scripts/perception_web_monitor/status.sh
bash scripts/perception_web_monitor/stop.sh
```

本机浏览器访问 `http://127.0.0.1:8088/`。服务仅绑定回环地址。
远程电脑执行 `ssh -N -L 8088:127.0.0.1:8088 honkanmae@<Orin-IP>` 后，在该电脑打开同一网址。
退出浏览器不停止监控进程；暂停按钮只暂停浏览器取图。

## 数据与显示

- 两路底图默认读取接收节点的本地 JPEG 通道（Linux abstract Unix stream socket），
  不再通过 DDS 订阅 1920×1080 raw Image。复用接收到的 JPEG，不增加发送侧编码。
  JPEG 头携带与 ROS Image/CameraInfo 完全相同的 sec/nanosec 时间戳。
  `monitor.json` 中 `image_transport: "ros_raw"` 可恢复原始图像订阅方式。
- 绿色框：`/rsu1/{gate,intersection}/yolo26s/objects`。
- 青色雷达框：`/perception/object_recognition/detection/centerpoint/objects`；默认显示投影后的二维外接矩形，可切换三维线框。
- 内参和畸变：对应相机 `camera_info`；外参：TF 中检测坐标系到相机光学坐标系的变换。
- 原始图带畸变，投影使用 K、D；图像和 CameraInfo 分辨率及 frame 必须匹配。
- 默认最大 10 FPS、1280×720（保持源比例）、JPEG 质量 78。
- `/api/status` 包含实际输入 FPS、输出 FPS、输入最大间隔、跳过帧数和 JPEG 连接错误。
  JPEG 发送在独立线程中进行，只缓存最新帧；慢网页不会阻塞相机 ROS 发布。
- 每个相机只保留 3 张图，等待约 80 ms 供检测到达，然后选择最新可处理帧；处理落后丢旧帧。
- 检测按 header 时间戳最近邻匹配，最大差 100 ms，超限显示 UNSYNC 且不绘制该层。
- 所有浏览器共享每路一次 JPEG 编码；慢客户端不会阻塞 ROS 接收。
- 直接投影 CenterPoint 八角点和十二条边，裁剪相机近裁面；二维模式取投影包围矩形再裁剪到图像，三维模式画裁剪后的线段。没有订阅点云或新增推理模型。
- 可选 ID 关联见 `scripts/ID_association/README.md`；关联在线时等待下限为140 ms，否则80 ms。绿色 YOLO 与紫色关联轨迹矩形不受青色框显示开关影响。
- 三维框可见数量不等于 YOLO 数量，两者本来就有视域与检测差异。

## 时间与精度说明

当前相机接收节点用解码后的本机时间戳，并非硬件曝光时间。因此页面 Δt 表示消息时间差，不能证明硬件级同步。
需要根据移动物体实际叠框验证时间偏差及外参；有 TF、内参不等于已经验证投影准确。
输出超过 2 秒未更新时页面标记过期，FPS 显示 0，不能把旧图当直播。
状态中的 image_age_ms 是 ROS 图像消息时间戳距处理时刻，不含后续编码、网络及浏览器显示延迟。

## 接口

- `/`：双路页面。
- `/stream/gate.mjpg`、`/stream/intersection.mjpg`：MJPEG。
- `/api/status`：各路同步状态、计数、输出频率和新鲜度。
- `GET /api/display`：当前 `lidar_box_mode`。`POST /api/display`，JSON `{"lidar_box_mode":"2d"}` 或 `{"lidar_box_mode":"3d"}`，立即切换后续画面。
- `/healthz`：HTTP 服务存活与各路状态；HTTP 200 不保证相机在线，请检查 online 字段。

独立启停，不会启动或停止 V2 检测节点。更改 JSON 后用 stop/start 重启监控。

## 2026-10-18 雷达显示切换

页面“雷达显示”下拉菜单可在 **2D 外接矩形 / 3D 线框** 间切换。设置对服务端两路共享视频生效，所有观看者看到同一模式，无需重启或刷新视频；已缓冲画面可能稍后才变化。暂停视频时先恢复视频才能看到变化。

默认来自 `config/perception_web_monitor/monitor.json` 的 `lidar_box_mode: "2d"`；菜单切换仅修改运行内存，服务重启恢复配置默认。显示模式也记录在每帧 `/api/status` 的 `lidar_box_mode`。

二维框仍来自原来的三维检测，不是 YOLO 框，不改变三维位置、尺寸、姿态、标定或 ID 匹配逻辑，也不会自动修正检测/投影偏差。与三维线框共用同一套近裁面处理及投影几何；在包围矩形与画面相交、但所有线框棱都位于画面外等边界场景，可见计数可能不同。

回归测试：`test_projection.py` 六项几何测试，`test_display_api.py` 两项接口测试（往返切换、非法请求不改变模式）。

## 2026-10-17 传输升级

主机 UDP 缓冲上限由 `/etc/sysctl.d/99-z-perception-transport.conf` 设置为 32 MiB；
CycloneDDS 为接收请求 16–32 MiB、发送请求 4–8 MiB，仅发现使用组播。
本机继续使用 CycloneDDS，没有套用 new-dt-server 的 FastDDS XML。

原始双雷达融合话题保持完整配对输出。CenterPoint 改用
`/sensing/lidar/centerpoint/pointcloud` 的最新点云入口，默认最多 12 Hz。
完整供帧后实测检测耗时 P95 约 83 ms，15 Hz 输入仍会积压；12 Hz 为此保留处理余量。
修改 `config/transport_overlay.env` 中 `PERCEPTION_CENTERPOINT_HZ` 后重启融合节点生效；
0 表示不限制。35 ms 双雷达配对门限未改变。

构建产物位于独立的 `install/transport_upgrade_20261017` overlay。
详细恢复点、测量记录及限制见 `plans/transport_upgrade_20261017/`。

协议与隔离测试位于 `scripts/perception_web_monitor/tests/`。
`scripts/perception_standard_v2/measure_transport.py` 可同时测量小话题、网页流和 UDP socket 丢包，
不额外订阅原始大图或点云。

## 2026-09-23 验证记录

容器在搭建时已经重启、感知节点未运行，已通过现有 V2 start 脚本启动完整检测链路。
监控使用单独的 CycloneDDS 配置：仅发现允许组播，图像数据单播。初版沿用数据组播时出现大幅降速，调整后相机发布日志恢复约 10 FPS。
实际分别读取 HTTP MJPEG 约 8 秒：Gate 8.27 FPS，Intersection 8.18 FPS；JPEG 解码尺寸均为 1280×720。
实时状态采样显示两路 YOLO/LiDAR 均可匹配 OK（YOLO 时间差 0 ms，LiDAR 约 -35 至 -12 ms）；部分帧仍可能显示 UNSYNC，明确跳过过期结果。
Gate 截图已验证青色三维投影框输出；夜间画面较暗，未据此认定外参精度或检测准确率已通过验证。
验证截图在 `logs/perception_web_monitor/validation_gate.jpg` 与 `validation_intersection.jpg`。
已检查 Python/Bash 语法、重复启动幂等性、HTTP 状态、JPEG 解码及仅回环监听；未做长期压力测试。
