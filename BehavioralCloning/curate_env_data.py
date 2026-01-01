# Data collected via FieldNavigators/collect_data is in full resolution. Instead of downsampling during
# training, run this script to compress all images into npz format.
# Also normalizes the steering to [-1, 1]

from pathlib import Path
import numpy as np
from PIL import Image
import tqdm

DATA_DIR = Path("../FieldNavigators/collect_data/build/total")
OUT_NPZ = Path("sao_paulo_dataset_mixed.npz")
SIZE = (96, 96)

rgb_list = []
act_list = []

pngs = sorted(DATA_DIR.glob("*.png"))

for png_path in tqdm.tqdm(pngs, desc="Processing images"):
    txt_path = png_path.with_suffix(".txt")
    if not txt_path.exists():
        continue

    # txt: throttle steering (we ignore throttle)
    line = txt_path.read_text().strip()
    if not line:
        continue
    throttle_str, steering_str = line.split()[:2]  # throttle ignored
    steering = float(steering_str) / 10.0  # [-10,10] -> [-1,1]

    img = Image.open(png_path).convert("RGB").resize(SIZE)
    rgb = np.asarray(img, dtype=np.uint8)  # (96,96,3)

    rgb_list.append(rgb)
    act_list.append([steering])

rgb = np.stack(rgb_list, axis=0).astype(np.uint8)  # (N,96,96,3)
act = np.asarray(act_list, dtype=np.float32)  # (N,1)

np.savez_compressed(OUT_NPZ, rgb=rgb, act=act)
print(f"Saved {OUT_NPZ} | rgb={rgb.shape} {rgb.dtype} | act={act.shape} {act.dtype}")


# Show a histogram of steering values
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.hist(act, bins=30, range=(-1.0, 1.0))
plt.title("Steering Value Distribution")
plt.xlabel("Steering Value")
plt.ylabel("Frequency")
plt.savefig("steering_histogram.png")
