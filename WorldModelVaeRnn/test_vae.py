# test_vae.py
import os, math
from pathlib import Path
import json
import torch
from torchvision import transforms
from PIL import Image
from types import SimpleNamespace
import cv2
import numpy as np

from train_vae import VAE


@torch.no_grad()
def show_dir_interactively(model, test_dir, device, img_size, window_name="VAE Recon"):
    model.eval()
    test_dir = Path(test_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        print(f"No images found in {test_dir}")
        return

    xform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1], CxHxW
        ]
    )

    for p in files:
        # load image
        img = Image.open(p).convert("RGB")
        x = xform(img).unsqueeze(0).to(device)  # 1xCxHxW

        # forward
        recon, _, _ = model(x)  # 1xCxHxW

        # to numpy uint8 for cv2
        def to_uint8(t):
            t = t.squeeze(0).clamp(0, 1).detach().cpu().numpy()  # CxHxW
            t = np.transpose(t, (1, 2, 0))  # HxWxC, RGB
            t = (t * 255.0).round().astype(np.uint8)
            return t

        src = to_uint8(x)
        rec = to_uint8(recon)

        # concat side-by-side, convert RGB->BGR for cv2
        pair_rgb = np.concatenate([src, rec], axis=1)
        pair_bgr = pair_rgb[:, :, ::-1]

        cv2.imshow(window_name, pair_bgr)
        cv2.setWindowTitle(window_name, f"{p.name}  |  left: input  right: recon")

        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord("q"):  # Esc or 'q' to quit early
            break

    cv2.destroyAllWindows()


# --------- load model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = json.load(open(Path("vae_outputs_32/args.json")))
state_dict = torch.load("vae_outputs_32/vae_state_43.pt", map_location="cpu")

model = VAE(
    img_size=model_config["image_size"],
    z_dim=model_config["latent_dim"],
    ch=model_config["base_ch"],
).to(device)
model.load_state_dict(state_dict)
model.eval()

# in case the decoder tracks image size differently
img_size = getattr(getattr(model, "dec", model), "img_size", model_config["image_size"])

# --------- args ----------
args = SimpleNamespace()
args.test_dir = "../FieldNavigators/collect_data/build/test_dir/"
args.image_size = img_size

# --------- run ----------
show_dir_interactively(model, args.test_dir, device, args.image_size)
