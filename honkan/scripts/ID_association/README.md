# ID association V1 — 双相机视觉框与三维轨迹关联

2026-10-18。初版是独立的、CPU 上运行的只读关联节点，不改变 CenterPoint、Tracker、YOLO、相机标定或点云前融合。没有新增 GPU 模型。

## 启停与查看

在宿主机执行（不需要重新编译）：

```bash
cd /mnt/SSD/workspace/tian_ws/autoware_runtime
bash scripts/ID_association/start.sh
bash scripts/perception_web_monitor/start.sh
```

前提：原有 V2 感知链路已经启动，包括两路 YOLO、Tracker、CameraInfo 和 TF。打开 http://127.0.0.1:8088/ 。远程查看仍需原来的 SSH 转发；不新增对外监听端口。

- 绿色框是 YOLO，`ID#数字` 是连续确认的三维轨迹关联，`ID#数字?` 是候选。
- 紫色矩形是已匹配的 Tracker 三维框投影外接矩形，用于检查几何对应。
- 青色雷达框仍来自原始 CenterPoint，并不是紫色框的数据源，因此两者可能不同。2026-10-18 起默认画二维外接矩形，网页“雷达显示”菜单可切回三维线框；该菜单不改变本关联算法。
- 左上角 `ID: OK 已确认数/关联数` 表示本帧处理状态与数量。OK 不等于身份正确率，也不等于所有目标都匹配。
- 未匹配的 YOLO 框不显示 ID；不凭空创建三维位置或全局身份。

仅停止关联：

```bash
bash scripts/ID_association/stop.sh
```

网页继续运行，后续新帧不再挂 ID；关联消息停止约两秒后恢复原来的等待时间。如果要恢复无网页后台的负载基线，也执行：

```bash
bash scripts/perception_web_monitor/stop.sh
```

关联没有加入 V2 总启动/停止脚本，需单独管理。修改 `config.json` 后重启关联节点；网页代码修改后重启网页。不要重复启动手工 Python 实例。

## 数据与身份

```text
CenterPoint → 现有 Tracker（UUID） ──┐
两路 CameraInfo + TF ───────────────┼→ ID_association → JSON ROS 话题 → 网页叠加
两路 YOLO（带源图像时间戳和 ROI） ──┘
```

输入：

- `/perception/object_recognition/tracking/objects`：TrackedObjects。
- `/rsu1/{gate,intersection}/yolo26s/objects`：DetectedObjectsWithFeature。
- `/rsu1/{gate,intersection}/camera_info`，以及 `/tf`、`/tf_static`。

输出 `/perception/id_association/results`，类型 `std_msgs/msg/String`，JSON 每路约 10 Hz。日志 `logs/ID_association/node.log`。

JSON 的 `uuid` 保留完整 Tracker UUID，是实际身份键；`display_id` 是节点内部统一分配的简短编号，两路共用同一映射。简短编号仅供显示，重启后重新编号；最多缓存 10000 个最近 UUID，极老轨迹重新出现时显示编号也可能变化。不能将显示编号当作跨重启永久身份。

每条结果包含 `camera`、精确的 `stamp_sec/stamp_nanosec`、状态、`track_dt_ms`、`matches`、`unmatched_yolo`、`ambiguous_yolo`、`projected_tracks`、`processing_ms`、`result_age_ms`。每个 match 包含 UUID、显示编号、YOLO 在原消息中的索引、两个矩形、代价、确认状态及连续次数。`cost` 是几何代价，不是概率。

## 算法步骤

1. YOLO 消息到达后等待至少 50 ms，给 Tracker 的近时刻结果留出到达机会；缓存有界。
2. 从 Tracker 缓存选择与 YOLO 图像时间戳最近的一帧，允许 ±150 ms；输入帧龄超过 600 ms 则拒绝。这里比较的是本机 ROS 时间戳，不代表硬件曝光/测距同步。
3. 用该轨迹时刻的 TF 将三维框转到相机坐标，使用当前 CameraInfo 的 K、D 投影。过滤相机后方/近裁面、畸变模型远外推、极小或异常大框，保守地拒绝跨近裁面的整个框。
4. 生成二维外接矩形；类别必须一致，CAR/TRUCK/BUS/TRAILER 允许互相兼容。YOLO 和轨迹的最低分数分别为 0.10 和 0.35。
5. 门限筛选：中心距离除以 YOLO 框对角线（下限15像素）后不得超过1；IoU不足0.01时，此归一化距离还须不超过0.5。
6. 代价 `0.65*(1-IoU) + 0.35*min(归一化中心距离,1)`；与上次关联的 YOLO 框 IoU>0.2 时减0.04。代价须小于0.85。
7. 带未匹配选项的匈牙利分配，每路每帧一对一。若任一竞争候选的代价优势不足0.08，拒绝该关联；不会为了满配强行分配。
8. 同一 UUID 连续三个已处理帧匹配、间隔不超过0.3秒且前后 YOLO 框 IoU>0.1，才标记 confirmed。漏检、歧义或时间/投影错误会重置确认。短时历史只参与代价，不会在当前无检测时补画旧 ID。
9. 网页以精确 YOLO 源时间戳及 ROI 索引查关联，绝不把前一帧的 ID 随意套在当前框上。没有同帧结果就不显示 ID。

网页在关联消息最近两秒内到达时，将帧等待下限从80 ms增加到140 ms，便于结果赶上叠框。通常增加约60 ms等待，实际帧龄取决于帧到达相位、调度和渲染；不是固定总延迟。停止关联后自动恢复80 ms。

## 边界与已知限制

- 这是几何关联基线，不是完整多模态融合；没有 ReID、跨断轨身份恢复、遮挡推理、脚点/地面约束或反向更新 Tracker。
- 当前未做速度外推：采用最近轨迹帧。需先验证速度坐标语义和质量，再加入预测。快速目标、时间偏差可能降低匹配质量。
- confirmed 只表示连续几何对应，不是确认物体真实存在。固定假目标若同时有视觉误检，也可能被连续关联。
- 三维框误差、投影偏差、人群交叉、重叠遮挡和 Tracker 本身换 ID 都会影响结果。歧义拒绝偏保守，可能经常出现候选或无 ID。
- 同 UUID 同时被两台相机关联是允许的；每个相机内部保持一对一。现场样本不保证出现跨相机重叠，不能据此宣称完成跨相机身份准确性验收。
- 不读取 raw image/点云、不占用额外 GPU 推理，也不会连接原始 JPEG IPC，避免挤掉网页的单客户端连接。网页本身的 CPU 开销仍然存在。

## 测试与现场记录

容器内使用现有 ROS 环境：

```bash
docker exec tian_autoware_runtime bash -lc 'source /autoware_runtime/scripts/perception_standard_v2/env.sh; cd /autoware_runtime/scripts/ID_association; python3 -m unittest -v test_core'
docker exec tian_autoware_runtime bash -lc 'source /autoware_runtime/scripts/perception_standard_v2/env.sh; python3 /autoware_runtime/scripts/ID_association/validate_live.py --seconds 30 --output /autoware_runtime/logs/ID_association/validation_manual.json'
```

单元测试8项通过，覆盖空输入、IoU、唯一对应、类别/距离拒绝、竞争歧义、一对一和确认重置。

首次30秒现场记录 `logs/ID_association/validation_20261018.json`：

| 相机 | 输出帧 | 频率 | 匹配次数 / 其中确认次数 | 不同 UUID | 平均关联处理 |
|---|---:|---:|---:|---:|---:|
| gate | 298 | 9.94 Hz | 373 / 275 | 6 | 7.45 ms |
| intersection | 299 | 9.94 Hz | 173 / 137 | 7 | 4.48 ms |

均为 OK，一对一重复分配帧为0。歧义拒绝计数分别12、9。匹配/确认次数是跨帧累计，不是人数；这些数字不代表准确率或召回率，需要人工标注视频进一步评估串 ID、误关联和漏关联。
