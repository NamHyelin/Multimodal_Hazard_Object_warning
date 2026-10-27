# Multimodal Hazard Object Warning

**Semantic Communication · LiDAR CenterPoint · Image-to-Text · Text-to-Image · Digital Twin · AR**

![Python](https://img.shields.io/badge/Python-262139?style=for-the-badge&logo=python&logoColor=F472B6)
![ROS](https://img.shields.io/badge/ROS-252341?style=for-the-badge&logo=ros&logoColor=A78BFA)
![LiDAR](https://img.shields.io/badge/LiDAR_CenterPoint-202C49?style=for-the-badge&logoColor=79B8FF)
![Multimodal AI](https://img.shields.io/badge/Multimodal_GenAI-262139?style=for-the-badge&logoColor=F472B6)
![Digital Twin](https://img.shields.io/badge/Digital_Twin-252341?style=for-the-badge&logoColor=A78BFA)

**Research testbed developed during visiting research at the Institute of Science Tokyo, Japan.**  
**Publication status:** Manuscript in preparation.

---

## Overview

This repository contains the implementation of a **real-world multimodal semantic communication testbed for contextual hazard awareness**. A roadside unit (RSU) equipped with a LiDAR sensor and camera detects objects in a road environment, while a remote digital twin (DT) server generates an augmented view for users of mobile AR glasses.

Instead of always transmitting raw images, the system can communicate **compact language-based semantic descriptions** of uncertain detections. The DT server uses those descriptions to generate **context-aware visual warning icons**, while recognized objects can be represented with predefined icons. The goal is to investigate how perception, generative AI, and semantic communication can support user-facing hazard visualization in practical deployments.

The architecture is designed to evaluate **end-to-end behavior and latency** in an operational RSU–DT–AR workflow. The presence of a generated warning does not by itself establish validated safety benefits; those require separate experiments.

### End-to-End Pipeline

```text
              ROADSIDE UNIT (RSU / honkan)
        Camera frames + LiDAR point clouds
                         │
                         ▼
                LiDAR CenterPoint
              3D object detections
                         │
                         ▼
             Confidence-based routing
                  threshold = 0.4
                  /            \
           confident          low confidence
               │                   │
               ▼                   ▼
       Class + position      Project 3D object
                            into camera frame
                            and crop object ROI
                                    │
                                    ▼
                          Image-to-Text (I2T)
                          context description
                  \                 /
                   \               /
                    ▼             ▼
                     UDP messages
                          │
                          ▼
                DIGITAL TWIN SERVER
                Incoming object metadata
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Known object class       Text description
             │                         │
       Predefined icon          Text-to-Image (T2I)
                                       │
                                 Generated icon
             └────────────┬────────────┘
                          ▼
                Place warning on DT view
                          │
                          ▼
                    AR visualization
```

> **Detection clarification:** The primary documented RSU pipeline uses **LiDAR CenterPoint**. Although a script named `yolox_florence_live.py` exists for an alternative pipeline, YOLO/YOLOX is **not** the model used for the main LiDAR demonstration.

---

## Hardware and Software

The code is organized around a deployed roadside sensing system and a separate DT computing server. Exact hardware specifications must be confirmed from the original testbed inventory or machine outputs; they are intentionally **not guessed** here.

| Component | Function | Hardware / specification |
| :-- | :-- | :-- |
| RSU camera | Live image stream for object cropping and contextual inference | **Model, resolution, FPS, interface: to be verified** |
| RSU LiDAR | 3D sensing for CenterPoint object detection | **Manufacturer, model, channel count, scan frequency, range: to be verified** |
| RSU compute / Jetson | Runs RSU-side software and sensing pipeline, as configured | **Jetson exact module (e.g. Orin family), RAM, GPU, JetPack: to be verified** |
| RSU host (`honkan`) | Autoware runtime / local semantic processing | **CPU, GPU, RAM, OS: to be verified; do not assume Jetson and host are the same device** |
| Digital Twin server | Receives UDP metadata and runs icon generation/rendering | **CPU, GPU, GPU VRAM, RAM, OS: to be verified** |
| Visualization | Digital twin view and AR user interface | **AR device model / rendering configuration: to be verified** |

**Relevant software:** Autoware runtime, ROS topics, Python virtual environments, Docker, LiDAR CenterPoint, image-to-text captioning, text-to-image generation, UDP communication, digital twin visualization.

<details>
<summary>How to collect accurate hardware specifications (for maintainers)</summary>

On Linux hosts, review `lscpu`, `free -h`, `lsblk`, `nvidia-smi` (when available), and `/etc/os-release`. On NVIDIA Jetson, review `/etc/nv_tegra_release`, `cat /proc/device-tree/model`, `sudo jetson_release` (if installed), and JetPack/version metadata. For the LiDAR and camera, use manufacturer model labels, device configuration, and ROS camera/point-cloud metadata; ROS topics alone do **not** reliably identify hardware models.

**Do not upload:** screenshots/logs containing internal IP addresses, personal usernames, VPN profiles, tokens, or passwords.

</details>

---

## Semantic Processing

### 1. LiDAR object detection and confidence routing

The RSU subscribes to the LiDAR point cloud and CenterPoint detection topics. The main runtime uses `centerpoint_florence_live.py` and a **confidence threshold of 0.4** to distinguish between detection-driven icon selection and additional semantic interpretation.

- **Confident detection:** send an available class label and location metadata for a predefined icon.
- **Low-confidence detection (`confidence <= 0.4` in the experiment description):** identify the corresponding camera region, crop an image containing the detected object, and apply an image-to-text model to extract a short semantic description.

The exact comparison operator at the boundary (`<` versus `<=`) must be confirmed from the code before making it a formal algorithm claim.

### 2. Camera projection and region-of-interest (ROI) cropping

The LiDAR pipeline receives camera intrinsics and a camera image alongside CenterPoint detections and the point cloud. Its `--allow_intrinsic_rescale` flag indicates support for resizing camera intrinsics when image and calibration dimensions differ.

The operational idea is to **associate a 3D LiDAR detection with its corresponding camera-image region** and crop that region before image-to-text inference. **The exact crop expansion factor, minimum crop size, clipping behavior, and projection calibration are implementation-specific and should be documented after inspecting the source configuration.** No unsupported crop size is asserted here.

### 3. Image-to-Text (I2T)

The RSU uses an image-to-text captioning component; the filename `centerpoint_florence_live.py` indicates a **Florence-oriented pipeline**. The exact Florence model identifier/checkpoint, prompt, precision, maximum output length, and inference device have not been verified from the source in this document. The output is a contextual text description for an uncertain or unfamiliar object.

### 4. UDP communication

The RSU can transmit object metadata or text descriptions, together with location information, to a receiver on the DT server over **UDP**. The provided example uses receiver port **8001**. Use an environment-specific destination address rather than hard-coding an internal host address into a public README.

### 5. Text-to-Image (T2I) and icon generation

On receipt of a **known object class**, the DT system displays a **predefined icon**. On receipt of a **text description**, it invokes a T2I model to create a visual warning symbol for the detected situation and places the result in the digital twin view.

The research pipeline has previously used **Sana Sprint (0.6B)** to generate minimal silhouette-style hazard icons, with a low inference-step setting (e.g. two steps) and post-processing to simplify icon backgrounds. **Confirm the actual model checkpoint, prompt/negative prompt, image size, steps, thresholds, and icon-cache settings in `dt_server.py` before treating those values as this repository's finalized configuration.**

---

## Getting Started

### Requirements

- Authorized access to both RSU and DT computing environments
- Running RSU sensors and Autoware ROS environment
- RSU Docker runtime and project virtual environment
- DT-side Python virtual environment and model dependencies
- Permitted network route and matching UDP destination/receiver port
- Model weights and any private calibration/configuration files provisioned separately

The commands below describe the **LiDAR CenterPoint + I2T → UDP → T2I** experiment, not the optional YOLOX branch. Replace all placeholders with values from your **private** deployment configuration.

### A. Start sensors on the RSU

Connect through the institution-approved private network and log in to the RSU by your normal authorized method. In the RSU's Autoware runtime directory:

```bash
cd <AUTOWARE_RUNTIME_DIR>
./scripts/start/start.sh
./scripts/start/start_camera_receiver.sh
./scripts/start/status.sh
```

The sensor startup scripts generally need to be run only once for a session. The experiment can then be restarted inside the existing container without restarting all sensors.

> The optional YOLOX launch script is **not required** for the CenterPoint-based experiment.

### B. Enter the RSU runtime and activate the environment

```bash
docker exec -it tian_autoware_runtime bash
source /autoware_runtime/scripts/enter_autoware_runtime_bash/container_env.sh
cd /autoware_runtime/hyelin
source .venv/bin/activate
```

**Optional: clear previous experiment outputs** (run only if you do not need to preserve them):

```bash
rm -f results/frames/*
rm -f results/florence_crops/*
rm -f results/results.csv
```

### C. Run CenterPoint + image-to-text, with UDP transmission

Provide the DT server destination **privately** when launching the RSU process. This command includes the ROS topics and flags used in the testbed:

```bash
export DT_RECEIVER_IP="<DT_SERVER_PRIVATE_IP>"
export DT_RECEIVER_PORT="8001"

python centerpoint_florence_live.py \
  --image_topic /rsu1/gate/image_raw \
  --camera_info_topic /rsu1/gate/camera_info \
  --pointcloud_topic /sensing/lidar/concatenated/pointcloud \
  --centerpoint_topic /perception/object_recognition/detection/centerpoint/objects \
  --confidence_threshold 0.4 \
  --save_crops \
  --warmup \
  --allow_intrinsic_rescale \
  --receiver_ip "$DT_RECEIVER_IP" \
  --receiver_port "$DT_RECEIVER_PORT" \
  --dt_world_frame map
```

**Argument guide**

| Argument | Purpose |
| :-- | :-- |
| `--image_topic` | RSU camera image stream |
| `--camera_info_topic` | Camera calibration/intrinsics metadata |
| `--pointcloud_topic` | LiDAR point-cloud input |
| `--centerpoint_topic` | CenterPoint detection output |
| `--confidence_threshold 0.4` | Low-confidence semantic-processing threshold |
| `--save_crops` | Save image crops for inspection |
| `--warmup` | Warm up the model/inference pipeline |
| `--allow_intrinsic_rescale` | Allow camera intrinsic scaling when applicable |
| `--receiver_ip`, `--receiver_port` | DT UDP destination |
| `--dt_world_frame map` | Target digital twin/world frame name |

**For future experiments:** keep the validated original intact and work from a controlled `centerpoint_florence_live_rev.py` copy. Reuse that development file rather than generating a new script for every change.

### D. Start the UDP receiver on the DT server (Terminal 1)

Connect to the DT host using authorized access, then run:

```bash
cd <DT_PROJECT_DIR>
source .venv/bin/activate
python receiver.py \
  --host 0.0.0.0 \
  --port 8001 \
  --exchange_dir "<DT_EXCHANGE_DIR>"
```

`0.0.0.0` binds the receiver to all available interfaces. Use appropriate firewall and VPN restrictions; do **not** expose the receiver openly to the internet.

### E. Start the digital twin application (Terminal 2)

```bash
cd <DT_PROJECT_DIR>
source .venv/bin/activate
python dt_server.py
```

With the RSU pipeline and the two DT-side processes running, the DT application can consume incoming detection/text data and generate or place icons for the visualization.

### F. Verify the experiment

1. Confirm camera frames, LiDAR point clouds, and CenterPoint detections are available on the expected ROS topics.
2. Confirm low-confidence objects result in saved image crops when `--save_crops` is enabled.
3. Confirm the RSU transmits packets to the approved DT receiver address and port.
4. Confirm the receiver updates the exchange directory and the DT process consumes the messages.
5. Inspect the DT visualization for predefined-class icons and newly generated text-conditioned icons.
6. Review the resulting frame files and CSV measurements, if enabled, for errors and latency analysis.

---

## Repository Structure

```text
Multimodal_Hazard_Object_warning/
├── README.md
├── honkan/              # RSU: sensing, CenterPoint/I2T pipeline
├── dt_server/           # DT: UDP receiver, T2I, visualization
└── environments/        # Environment and setup files
```

The folders reflect the published repository layout. Specific source paths and entry-point placement should be verified against the repository before running commands from its checkout.

---

## Research Directions

- **Communication-efficient semantic information:** transmit compact object labels or generated text rather than raw images for every event.
- **Uncertain object interpretation:** enrich low-confidence detections with contextual captions.
- **Adaptive visual alerts:** generate warnings for objects or situations not covered by the predefined icon set.
- **End-to-end testbed validation:** measure inference, transport, rendering, and user-facing response time separately.
- **Reproducibility:** compare conventional class-based warning overlays against text-conditioned generated warnings under controlled scenarios.

**Publication:** Research manuscript in preparation. Performance or safety improvements should be supported by measured experimental results before publication.

---

## Attribution and Notes

Developed during visiting research at the **Institute of Science Tokyo**. This README describes the experiment configuration provided by the project author. Hardware inventories, model checkpoints, image-crop constants, and device-specific optimizations require source/configuration verification before they can be published as precise specifications.

**Security:** No passwords, user identities, internal host addresses, named VPN profiles, or private server directories are included. Check committed scripts, `.env` files, configuration files, shell history, and Git history for sensitive data before publishing. If a credential was committed, rotate it; deleting it from the latest README does not remove it from Git history.
