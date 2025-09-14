import json
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from train_vae import VAE
import re
from pathlib import Path


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(x):
    s = x.name if isinstance(x, Path) else str(x)
    return [atoi(c) for c in re.split(r"(\d+)", s)]


@torch.no_grad()
def build_latent_dataset(img_dir, vae_ckpt, out_path, device="cuda"):
    img_dir = Path(img_dir)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # load VAE
    model_config = json.load(open(Path(vae_ckpt).with_name("args.json")))
    state_dict = torch.load(vae_ckpt, map_location="cpu")
    vae = VAE(
        img_size=model_config["image_size"],
        z_dim=model_config["latent_dim"],
        ch=model_config["base_ch"],
    ).to(device)
    vae.load_state_dict(state_dict)
    vae.eval()

    # tfm same as sample_vae
    tfm = transforms.Compose(
        [
            transforms.Resize((model_config["image_size"], model_config["image_size"])),
            transforms.ToTensor(),
        ]
    )

    img_files = sorted(Path(img_dir).glob("*.png"), key=natural_keys)
    latent_vecs, actions, img_paths = [], [], []

    for img_path in tqdm(img_files, desc="Encoding images"):
        stem = img_path.stem
        print(stem)
        action_txt_path = img_dir / f"{stem}.txt"
        if not action_txt_path.exists():
            assert False, f"Missing action file for {img_path}: {action_txt_path}"

        # load image and encode
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)
        z, mu, logvar = vae.encode(x)  # use mu for deterministic latent
        latent_vecs.append(mu.squeeze(0).cpu().numpy())

        # read action
        with open(action_txt_path, "r") as f:
            line = f.readline().strip()
            throttle, steer = map(float, line.split())
            actions.append([throttle, steer])

        img_paths.append(str(img_path))

    latent_vecs = np.stack(latent_vecs, axis=0)  # (N, d_z)
    actions = np.stack(actions, 0)  # (N, 2)

    # store paths as fixed-length unicode (avoids pickle)
    maxlen = max(len(p) for p in img_paths) if img_paths else 1
    paths_arr = np.array(img_paths, dtype=f"<U{maxlen}")

    np.savez_compressed(out_path, z=latent_vecs, a=actions, paths=paths_arr)
    print(f"Saved {latent_vecs.shape[0]} samples to {out_path}")


if __name__ == "__main__":
    build_latent_dataset(
        img_dir="../FieldNavigators/collect_data/build/train_dir/measurements_and_actions/",
        vae_ckpt="vae_outputs/vae_state.pt",
        out_path="latent_dataset.npz",
        device="cuda",
    )
