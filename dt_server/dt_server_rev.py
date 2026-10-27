from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from diffusers import SanaSprintPipeline
from PIL import Image

BASE_DIR = Path('/home/morita/hyelin')
ICON_DIR = BASE_DIR / 'icons'
BACKGROUND_IMAGE_PATH = BASE_DIR / 'AR_background.jpg'

RUN_TAG = os.environ.get('SEMCOMM_RUN_ID') or datetime.now().astimezone().strftime('%Y%m%d_%H%M')
RUN_DIR = BASE_DIR / 'results' / f'results_{RUN_TAG}'
EXCHANGE_DIR = BASE_DIR / 'dt_exchange'
LATEST_JSON_PATH = EXCHANGE_DIR / 'latest.json'
GENERATED_ICON_DIR = RUN_DIR / 'generated_icons'
RENDERED_DIR = RUN_DIR / 'rendered_frames'
LOG_DIR = RUN_DIR / 'logs'
PROCESSING_CSV_PATH = LOG_DIR / 'processing_log.csv'

RENDER_WIDTH = 2048
RENDER_HEIGHT = 1152
JPEG_QUALITY = 90
POLL_INTERVAL_SEC = 0.02

MIN_ICON_PX = 28
MAX_ICON_PX = 420
WHITE_THRESHOLD = 245

# =========================================================
# Manual map -> background calibration
# =========================================================
# These values come directly from the guide lines drawn on
# AR_background copy.jpg (2048 x 1152).
#
# Pink line: map X axis
#   left  end: x = -13486
#   right end: x = -13438
#
# Yellow ground guide: map Y direction
#   left end:  y = -43709
#   right end: y = -43716
#
# Green line: absolute map Z guide
#   lower end: z = 0
#   upper end: z = 71.3
#
# Pixel positions were estimated from the supplied 2048x1152 annotated image.
CAL_IMAGE_W = 2048
CAL_IMAGE_H = 1152

# X / pink guide.
MAP_X_LEFT = -13486.0
MAP_X_RIGHT = -13438.0
PINK_LEFT_PX = np.array([0.0, 824.0], dtype=np.float64)
PINK_RIGHT_PX = np.array([2041.0, 824.0], dtype=np.float64)

# Y guide (user-updated).
# In the annotated background, the green vertical guide represents map-Y:
#   top    -> y = -43712
#   bottom -> y = -43720
# We use this as the primary front/back placement on the background.
MAP_Y_TOP = -43700.0 #-43712.0
MAP_Y_BOTTOM = -43730.0 #-43720.0
Y_TOP_PX = np.array([1310.0, 726.0], dtype=np.float64)
Y_BOTTOM_PX = np.array([1310.0, 824.0], dtype=np.float64)

# Z guide (user-updated).
# The blue vertical guide shows the apparent image height for z.
# Use z=0 on the pink baseline and z=71 at the upper blue endpoint.
MAP_Z_BOTTOM = 0.0
MAP_Z_TOP = 71.0
Z_BOTTOM_PX = np.array([1153.0, 824.0], dtype=np.float64)
Z_TOP_PX = np.array([1153.0, 755.0], dtype=np.float64) #[1153.0, 726.0] 뒤에꺼 키우면 밑으로 내려옴

# Icon size is estimated from the local ground-plane pixel-per-meter scale.
# Keep the previous gain because the user said overall scale became much better.
ICON_SCALE_GAIN = 2.5 #1.35

# Allow a small amount outside the annotated X/Y range before rejecting.
MAP_CAL_MARGIN_PX = 120

# Optional calibration diagnostics.
DEBUG_DRAW_MAP_POINT = False

# Pre-computed map-to-pixel guide vectors.
_X_PX_PER_M = (
    (PINK_RIGHT_PX[0] - PINK_LEFT_PX[0])
    / (MAP_X_RIGHT - MAP_X_LEFT)
)

# Front/back motion on the background image.
_Y_PX_PER_M = (
    (Y_TOP_PX[1] - Y_BOTTOM_PX[1])
    / (MAP_Y_TOP - MAP_Y_BOTTOM)
)

# Vertical lift per meter of z.
_Z_PX_PER_M = (
    (Z_TOP_PX[1] - Z_BOTTOM_PX[1])
    / (MAP_Z_TOP - MAP_Z_BOTTOM)
)
MODEL_ID = 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers'
HEIGHT = 448
WIDTH = 448
GUIDANCE_SCALE = 4.5
MAX_SEQUENCE_LENGTH = 64
NUM_INFERENCE_STEPS = 1
INTERMEDIATE_TIMESTEPS = 1.3 if NUM_INFERENCE_STEPS == 2 else None
PROMPT_PREFIX = (
    'single isolated object icon, centered, one object only, '
    'solid black silhouette, pure white background, '
    'no scenery, no road, no building, no text. '
    'Object description: '
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for d in [EXCHANGE_DIR, ICON_DIR, GENERATED_ICON_DIR, RENDERED_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def append_csv_row(row: dict) -> None:
    fieldnames = [
        'frame_index', 'latest_only_seq', 'object_count',
        'processing_started_at_utc', 'processing_finished_at_utc',
        'processing_sec', 'sana_icon_sec', 'composition_sec', 'image_save_sec',
        'saved_image_path', 'status', 'error'
    ]
    exists = PROCESSING_CSV_PATH.exists()
    with PROCESSING_CSV_PATH.open('a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def make_safe_name(text: str, max_length: int = 80) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in text.strip())
    safe = '_'.join(p for p in safe.split('_') if p)
    return (safe or 'object')[:max_length]


def caption_to_cache_name(caption: str) -> str:
    digest = hashlib.md5(caption.encode('utf-8')).hexdigest()[:10]
    # v2: generated image keeps its original foreground appearance;
    # only the white background is removed.
    return f'{make_safe_name(caption, 40)}_{digest}_bgremoved_v2.png'


def remove_white_background(image: Image.Image, force_black: bool = True) -> Image.Image:
    arr = np.array(image.convert('RGBA'), dtype=np.uint8)
    white = np.all(arr[:, :, :3] >= WHITE_THRESHOLD, axis=2)
    arr[white, 3] = 0
    if force_black:
        fg = arr[:, :, 3] > 0
        arr[fg, :3] = 0
    out = Image.fromarray(arr, mode='RGBA')
    bbox = out.getbbox()
    return out.crop(bbox) if bbox is not None else out


def paste_icon_centered(canvas: Image.Image, icon: Image.Image, center_xy: tuple[int, int], box_wh: tuple[int, int]) -> None:
    icon = icon.convert('RGBA').copy()
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    icon.thumbnail((max(1, int(box_wh[0])), max(1, int(box_wh[1]))), resampling)
    x = int(center_xy[0] - icon.width / 2)
    y = int(center_xy[1] - icon.height / 2)
    canvas.alpha_composite(icon, dest=(x, y))


def class_icon_candidates(class_name: str) -> list[str]:
    cls = (class_name or '').strip().upper()
    alias = {
        'UNKNOWN': ['UNKNOWN'], 'CAR': ['CAR'], 'TRUCK': ['TRUCK'], 'BUS': ['BUS'],
        'TRAILER': ['TRAILER'], 'MOTORCYCLE': ['motorcycle', 'MOTORCYCLE'],
        'BICYCLE': ['BICYCLE', 'bike'], 'PEDESTRIAN': ['PEDESTRIAN', 'person'],
        'PERSON': ['PERSON', 'pedestrian'],
    }
    return alias.get(cls, [cls.lower()])


def load_class_icon(class_name: str) -> Optional[Image.Image]:
    for stem in class_icon_candidates(class_name):
        for ext in ['.png', '.webp', '.jpg', '.jpeg']:
            path = ICON_DIR / f'{stem}{ext}'
            if path.exists():
                return remove_white_background(Image.open(path).convert('RGBA'), force_black=False)
    return None


def _map_x_to_pink_u(map_x: float) -> float:
    """Map absolute map X to the pink-guide pixel coordinate."""
    alpha = (float(map_x) - MAP_X_LEFT) / (MAP_X_RIGHT - MAP_X_LEFT)
    return float(
        PINK_LEFT_PX[0]
        + alpha * (PINK_RIGHT_PX[0] - PINK_LEFT_PX[0])
    )


def project_world_to_view2(
    world_position_m,
    image_width: int,
    image_height: int,
):
    """
    Empirical map-coordinate projection derived from the user's latest manual
    background calibration.

    Mapping used now:
      - map X : pink horizontal baseline (left=-13486, right=-13438)
      - map Y : green vertical guide    (top=-43712, bottom=-43720)
      - map Z : blue vertical height    (top=71, bottom=0)

    The important change is that map-Y is now treated as a mostly vertical
    front/back placement on the background, which matches the user's new
    annotation better than the previous slanted yellow-line model.

    Returns:
      (u, v, x_px_per_m, y_px_per_m, local_px_per_m)
    """
    if world_position_m is None or len(world_position_m) < 3:
        return None

    map_x = float(world_position_m[0])
    map_y = float(world_position_m[1])
    map_z = float(world_position_m[2])

    if not np.isfinite([map_x, map_y, map_z]).all():
        return None

    # 1) map X -> horizontal placement along the pink baseline.
    u_base = _map_x_to_pink_u(map_x)

    # 2) map Y -> front/back placement. Higher y (-43712) appears higher.
    y_alpha = (map_y - MAP_Y_BOTTOM) / (MAP_Y_TOP - MAP_Y_BOTTOM)
    v_ground = Y_BOTTOM_PX[1] + y_alpha * (Y_TOP_PX[1] - Y_BOTTOM_PX[1])

    # 3) map Z -> lift upward from the local ground position.
    v = v_ground + (map_z - MAP_Z_BOTTOM) * _Z_PX_PER_M
    u = u_base

    # Calibration was measured at 2048x1152. Scale if render resolution changes.
    sx_img = float(image_width) / float(CAL_IMAGE_W)
    sy_img = float(image_height) / float(CAL_IMAGE_H)

    u *= sx_img
    v *= sy_img

    x_px_per_m = abs(_X_PX_PER_M) * sx_img
    y_px_per_m = abs(_Y_PX_PER_M) * sy_img
    local_px_per_m = 0.5 * (x_px_per_m + y_px_per_m)

    return (
        int(round(u)),
        int(round(v)),
        float(x_px_per_m),
        float(y_px_per_m),
        float(local_px_per_m),
    )


def scaled_icon_box_px(
    size_m: dict,
    x_px_per_m: float,
    y_px_per_m: float,
    local_px_per_m: float,
) -> tuple[int, int]:
    """
    Estimate icon size from the manually calibrated map scale.

    Length/width use the horizontal ground-plane scales.
    Height uses the local average ground-plane scale rather than the green
    absolute-altitude guide: the green guide is for POSITION, not for the
    apparent physical height of a person/car icon.
    """
    length_m = max(0.05, float(size_m.get("length", 1.0)))
    width_m = max(0.05, float(size_m.get("width", 1.0)))
    height_m = max(0.05, float(size_m.get("height", 1.0)))

    box_w = int(round(
        ICON_SCALE_GAIN
        * max(
            length_m * x_px_per_m,
            width_m * y_px_per_m,
        )
    ))
    box_h = int(round(
        ICON_SCALE_GAIN
        * height_m
        * local_px_per_m
    ))

    return (
        max(MIN_ICON_PX, min(MAX_ICON_PX, box_w)),
        max(MIN_ICON_PX, min(MAX_ICON_PX, box_h)),
    )


class SanaIconGenerator:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU를 찾을 수 없습니다.')
        self.device = 'cuda'
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print('Loading SANA-Sprint...', flush=True)
        self.pipe = SanaSprintPipeline.from_pretrained(MODEL_ID, torch_dtype=self.dtype)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.disable_slicing()
        self.pipe.vae.disable_tiling()
        with torch.inference_mode():
            _ = self.pipe(
                **self._kwargs(
                    'single isolated object icon, centered, one object only, '
                    'solid black silhouette, pure white background, no text',
                    0,
                )
            ).images[0]
        torch.cuda.synchronize()
        print('SANA-Sprint ready.', flush=True)

    def _kwargs(self, prompt: str, seed: int) -> dict:
        d = dict(
            prompt=prompt, height=HEIGHT, width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS, intermediate_timesteps=INTERMEDIATE_TIMESTEPS,
            guidance_scale=GUIDANCE_SCALE, use_resolution_binning=False,
            max_sequence_length=MAX_SEQUENCE_LENGTH, clean_caption=False,
            complex_human_instruction=[],
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        if INTERMEDIATE_TIMESTEPS is not None:
            d['intermediate_timesteps'] = INTERMEDIATE_TIMESTEPS
        return d

    def generate_icon(self, caption: str) -> Image.Image:
        cache_path = GENERATED_ICON_DIR / caption_to_cache_name(caption)
        if cache_path.exists():
            return Image.open(cache_path).convert('RGBA')
        t0 = time.perf_counter()
        with torch.inference_mode():
            image = self.pipe(**self._kwargs(PROMPT_PREFIX + ' '.join(caption.split()), 42)).images[0]
        torch.cuda.synchronize()
        print(f'[SANA] {time.perf_counter()-t0:.3f}s caption={caption!r}', flush=True)
        # IMPORTANT:
        # Keep the generated foreground pixels exactly as SANA produced them.
        # Only remove the white background and save as transparent PNG.
        icon = remove_white_background(image, force_black=False)
        icon.save(cache_path, format='PNG')
        return icon


class LatestFileRenderer:
    def __init__(self) -> None:
        ensure_dirs()
        if not BACKGROUND_IMAGE_PATH.exists():
            raise FileNotFoundError(f'Background image not found: {BACKGROUND_IMAGE_PATH}')
        bg = Image.open(BACKGROUND_IMAGE_PATH).convert('RGBA')
        resampling = getattr(Image, 'Resampling', Image).LANCZOS
        if bg.size != (RENDER_WIDTH, RENDER_HEIGHT):
            bg = bg.resize((RENDER_WIDTH, RENDER_HEIGHT), resampling)
        self.background = bg
        self.class_icons: dict[str, Optional[Image.Image]] = {}
        for cls in ['UNKNOWN','CAR','TRUCK','BUS','TRAILER','MOTORCYCLE','BICYCLE','PEDESTRIAN']:
            icon = load_class_icon(cls)
            self.class_icons[cls] = icon
            print(f'[ICON {"READY" if icon is not None else "MISSING"}] class={cls}', flush=True)
        self.generator = SanaIconGenerator()
        self.last_processed_seq = -1
        print(f'[RENDERER READY] watching={LATEST_JSON_PATH}', flush=True)
        print(f'[RENDERER READY] output={RENDERED_DIR}', flush=True)

    def read_latest(self) -> Optional[dict]:
        if not LATEST_JSON_PATH.exists():
            return None
        try:
            return json.loads(LATEST_JSON_PATH.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def process_payload(self, payload: dict) -> Path:
        frame = int(payload['frame_index'])
        seq = int(payload['latest_only_seq'])
        objects = payload.get('objects', [])
        with (LOG_DIR / 'payloads.jsonl').open('a') as f:
            f.write(json.dumps(dict(observed_at=time.time(), payload=payload)) + '\n')
        started = utc_now_iso()
        t0 = time.perf_counter()
        sana_sec = 0.0
        canvas = self.background.copy()
        image_width, image_height = canvas.size

        for obj in objects:
            cls = str(obj.get('class', '')).strip().upper()
            caption = obj.get('florence_caption')
            world_pos = obj.get('world_position_m')
            size_m = obj.get('size_m') or {}
            projected = project_world_to_view2(world_pos, image_width, image_height)
            if projected is None:
                print(
                    f'[OVERLAY SKIP] frame={frame} class={cls} '
                    f'world={world_pos} reason=invalid_map_position',
                    flush=True,
                )
                continue

            u, v, x_scale, y_scale, local_scale = projected

            if not (
                -MAP_CAL_MARGIN_PX <= u < image_width + MAP_CAL_MARGIN_PX
                and -MAP_CAL_MARGIN_PX <= v < image_height + MAP_CAL_MARGIN_PX
            ):
                print(
                    f'[OVERLAY SKIP] frame={frame} class={cls} '
                    f'world={world_pos} px=({u},{v}) reason=outside_calibrated_view',
                    flush=True,
                )
                continue

            # Clamp a small near-edge extrapolation into the visible image.
            u = min(max(u, 0), image_width - 1)
            v = min(max(v, 0), image_height - 1)

            box_w, box_h = scaled_icon_box_px(
                size_m,
                x_px_per_m=x_scale,
                y_px_per_m=y_scale,
                local_px_per_m=local_scale,
            )
            class_icon = self.class_icons.get(cls)
            if cls not in self.class_icons:
                class_icon = load_class_icon(cls)
                self.class_icons[cls] = class_icon

            if caption is not None and str(caption).strip():
                # Florence/T2I case:
                # use ONLY the newly generated transparent PNG.
                # Do not overlay the predefined detection-class icon.
                ts = time.perf_counter()
                clean_caption = str(caption).strip()
                cache_hit = (GENERATED_ICON_DIR / caption_to_cache_name(clean_caption)).exists()
                gen_icon = self.generator.generate_icon(clean_caption)
                with (LOG_DIR / 'generation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(frame_index=frame, seq=seq,
                        object_id=obj.get('object_id'), caption=clean_caption,
                        cache_hit=cache_hit, service_sec=time.perf_counter()-ts)) + '\n')
                sana_sec += time.perf_counter() - ts
                paste_icon_centered(
                    canvas,
                    gen_icon,
                    (u, v),
                    (box_w, box_h),
                )
            elif class_icon is not None:
                # Normal CenterPoint case without T2I:
                # keep using the predefined class icon.
                paste_icon_centered(canvas, class_icon, (u, v), (box_w, box_h))
            else:
                print(f'[ICON NOT DRAWN] frame={frame} class={cls} reason=missing_icon', flush=True)

            print(
                f'[OVERLAY] frame={frame} seq={seq} class={cls} '
                f'world={world_pos} px=({u},{v}) '
                f'scale_xy=({x_scale:.1f},{y_scale:.1f})px/m '
                f'icon={box_w}x{box_h}px',
                flush=True,
            )

        save_path = RENDERED_DIR / f'frame_{frame:06d}_seq_{seq:09d}.jpg'
        ts = time.perf_counter()
        canvas.convert('RGB').save(save_path, format='JPEG', quality=JPEG_QUALITY, subsampling=2, optimize=False)
        save_sec = time.perf_counter() - ts
        total = time.perf_counter() - t0
        append_csv_row({
            'frame_index': frame, 'latest_only_seq': seq, 'object_count': len(objects),
            'processing_started_at_utc': started, 'processing_finished_at_utc': utc_now_iso(),
            'processing_sec': f'{total:.6f}', 'sana_icon_sec': f'{sana_sec:.6f}',
            'composition_sec': '', 'image_save_sec': f'{save_sec:.6f}',
            'saved_image_path': str(save_path), 'status': 'processed', 'error': ''
        })
        print(f'[PROCESS DONE] frame={frame} seq={seq} total={total:.3f}s sana={sana_sec:.3f}s save={save_sec:.3f}s', flush=True)
        return save_path

    def run_forever(self) -> None:
        while True:
            payload = self.read_latest()
            if payload is not None:
                try:
                    seq = int(payload.get('latest_only_seq', -1))
                except Exception:
                    seq = -1
                if seq > self.last_processed_seq:
                    try:
                        self.process_payload(payload)
                    except Exception as exc:
                        print(f'[PROCESS ERROR] seq={seq}: {exc}', flush=True)
                    finally:
                        self.last_processed_seq = seq
                    continue
            time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    renderer = LatestFileRenderer()
    try:
        renderer.run_forever()
    except KeyboardInterrupt:
        print('\n[RENDERER] stopped.', flush=True)


def run_instrumented_receiver():
    # Same receiver implementation/protocol, with measurement at callback entry.
    import receiver
    ensure_dirs()
    original = receiver.LatestUDPProtocol.datagram_received
    def measured(protocol, data, addr):
        received_wall = time.time()
        original(protocol, data, addr)
        try:
            payload = json.loads(data)
            with (LOG_DIR / 'receptions.jsonl').open('a') as f:
                f.write(json.dumps(dict(received_wall=received_wall,
                    bytes=len(data), payload=payload)) + '\n')
        except Exception as exc:
            print('[TELEMETRY ERROR]', exc, flush=True)
    receiver.LatestUDPProtocol.datagram_received = measured
    sys.argv.remove('--receiver')
    receiver.main()

if __name__ == '__main__':
    if '--receiver' in sys.argv:
        run_instrumented_receiver()
    else:
        main()
