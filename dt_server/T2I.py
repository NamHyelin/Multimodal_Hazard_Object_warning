from pathlib import Path
import time
import torch
from diffusers import SanaSprintPipeline

MODEL_ID = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
OUTPUT_PATH = Path.home() / "hyelin" / "sana_test_car.png"

PROMPT = (
    "single isolated object icon, centered, one object only, "
    "solid black silhouette, pure white background, "
    "no scenery, no road, no building, no text. "
    "Object description: car facing left"
)

HEIGHT = 448
WIDTH = 448
GUIDANCE_SCALE = 4.5
MAX_SEQUENCE_LENGTH = 64
NUM_INFERENCE_STEPS = 1
INTERMEDIATE_TIMESTEPS = 1.3 if NUM_INFERENCE_STEPS == 2 else None

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")

device = "cuda"

if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16

print("Loading model...")
print("device:", device)
print("dtype :", dtype)

pipe = SanaSprintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)

# file 코드와 유사하게
pipe.vae.disable_slicing()
pipe.vae.disable_tiling()

# warm-up 1회
print("Warm-up...")
with torch.inference_mode():
    _ = pipe(
        prompt="single isolated object icon, centered, one object only, solid black silhouette, pure white background, no text",
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        use_resolution_binning=False,
        intermediate_timesteps=INTERMEDIATE_TIMESTEPS,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        clean_caption=False,
        complex_human_instruction=[],
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]

torch.cuda.synchronize()

print("Generating...")
start = time.perf_counter()

kwargs = dict(
    prompt=PROMPT,
    height=HEIGHT,
    width=WIDTH,
    num_inference_steps=NUM_INFERENCE_STEPS,
    intermediate_timesteps=INTERMEDIATE_TIMESTEPS,
    guidance_scale=GUIDANCE_SCALE,
    use_resolution_binning=False,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    clean_caption=False,
    complex_human_instruction=[],
    generator=torch.Generator(device=device).manual_seed(42),
)

if INTERMEDIATE_TIMESTEPS is not None:
    kwargs["intermediate_timesteps"] = INTERMEDIATE_TIMESTEPS

with torch.inference_mode():
    image = pipe(**kwargs).images[0]

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

image.save(OUTPUT_PATH)

print(f"Saved: {OUTPUT_PATH}")
print(f"Inference time: {elapsed:.3f} sec")
