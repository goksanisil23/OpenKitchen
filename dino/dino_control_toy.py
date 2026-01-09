import math
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
IMG = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# Dataset / expert
N_SAMPLES = 20000
HORIZON = 30
KP = 1.2
ACTION_CLIP = 0.12
GOAL_THRESH = 0.04

# Train
BATCH = 256
EPOCHS = 15
LR = 2e-3

DINO_NAME = "vit_small_patch16_224.dino"


# -----------------------------
# Toy env renderer
# -----------------------------
def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sample_pos(rng: np.random.RandomState) -> np.ndarray:
    return rng.uniform(0.1, 0.9, size=(2,)).astype(np.float32)


def expert_action(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    a = KP * (goal - pos)
    a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
    return a.astype(np.float32)


def step(pos: np.ndarray, a: np.ndarray) -> np.ndarray:
    pos = pos + a
    pos[0] = clip01(float(pos[0]))
    pos[1] = clip01(float(pos[1]))
    return pos.astype(np.float32)


def render(pos: np.ndarray, goal: np.ndarray, img_size: int = IMG) -> np.ndarray:
    # plain background, deterministic
    img = Image.new("RGB", (img_size, img_size), (20, 20, 20))
    d = ImageDraw.Draw(img)

    def to_px(p):
        return int(p[0] * (img_size - 1)), int(p[1] * (img_size - 1))

    # goal: green ring
    gx, gy = to_px(goal)
    gr = 6
    d.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), outline=(0, 255, 0), width=2)

    # agent: red filled circle
    ax, ay = to_px(pos)
    ar = 4
    d.ellipse((ax - ar, ay - ar, ax + ar, ay + ar), fill=(255, 0, 0))

    return np.asarray(img, dtype=np.uint8)


# -----------------------------
# Dataset (expert rollouts -> (image, action))
# -----------------------------
class ExpertDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.images = np.zeros((n_samples, IMG, IMG, 3), dtype=np.uint8)
        self.actions = np.zeros((n_samples, 2), dtype=np.float32)
        self._gen(n_samples)

    def _gen(self, n_samples: int):
        i = 0
        while i < n_samples:
            pos = sample_pos(self.rng)
            goal = sample_pos(self.rng)
            for _ in range(HORIZON):
                a = expert_action(pos, goal)
                self.images[i] = render(pos, goal)
                self.actions[i] = a
                i += 1
                pos = step(pos, a)
                if i >= n_samples:
                    break
                if np.linalg.norm(goal - pos) < GOAL_THRESH:
                    break

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0  # CHW
        a = torch.from_numpy(self.actions[idx]).float()
        return x, a


# -----------------------------
# DINO encoder (frozen) + policy head
# -----------------------------
def load_dino_encoder(name: str, device: str):
    import timm

    m = timm.create_model(name, pretrained=True)
    m.eval().to(device)

    if not hasattr(m, "forward_features"):
        raise RuntimeError(f"{name} has no forward_features() in timm.")

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode(x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] in [0,1]
        # DINOv2 ViTs typically expect 518x518
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = (x - mean) / std
        feats = m.forward_features(x)

        # timm models vary in output structure
        if isinstance(feats, dict):
            if "cls_token" in feats:
                z = feats["cls_token"]  # [B, D]
            elif "x" in feats:
                z = feats["x"][:, 0]  # [B, D] (CLS token)
            else:
                z = next(iter(feats.values()))
                if z.ndim == 3:
                    z = z[:, 0]
        else:
            z = feats
            if z.ndim == 3:
                z = z[:, 0]

        return F.normalize(z, dim=-1)

    # infer embedding dimension
    with torch.no_grad():
        dummy = torch.zeros(2, 3, IMG, IMG, device=device)
        emb_dim = encode(dummy).shape[-1]

    return encode, emb_dim


class PolicyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, z):
        return self.net(z)


# -----------------------------
# Train / Eval
# -----------------------------
def train_bc(encoder, policy, loader):
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    for ep in range(EPOCHS):
        total = 0.0
        n = 0
        for x, a in loader:
            x = x.to(DEVICE)
            a = a.to(DEVICE)

            with torch.no_grad():
                z = encoder(x)

            pred = policy(z)
            loss = F.mse_loss(pred, a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        print(f"epoch {ep+1}/{EPOCHS} mse={total/n:.6f}")


@torch.no_grad()
def rollout_eval(encoder, policy, n_episodes: int = 200):
    policy.eval()
    rng = np.random.RandomState(SEED + 123)
    success = 0
    steps_list = []

    for _ in range(n_episodes):
        pos = sample_pos(rng)
        goal = sample_pos(rng)

        for t in range(HORIZON):
            img = render(pos, goal)
            x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x = x.to(DEVICE)

            z = encoder(x)
            a = policy(z).squeeze(0).cpu().numpy().astype(np.float32)
            a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)

            pos = step(pos, a)
            if np.linalg.norm(goal - pos) < GOAL_THRESH:
                success += 1
                steps_list.append(t + 1)
                break

    sr = success / n_episodes
    avg_steps = float(np.mean(steps_list)) if steps_list else float("nan")
    print(f"rollout eval: success_rate={sr:.3f}, avg_steps_to_goal={avg_steps:.2f}")


def render_pil(pos: np.ndarray, goal: np.ndarray) -> Image.Image:
    """Same as render(), but returns a PIL image for easy annotation/saving."""
    img = Image.new("RGB", (IMG, IMG), (20, 20, 20))
    d = ImageDraw.Draw(img)

    def to_px(p):
        return int(p[0] * (IMG - 1)), int(p[1] * (IMG - 1))

    gx, gy = to_px(goal)
    gr = 6
    d.ellipse((gx - gr, gy - gr, gx + gr, gy + gr), outline=(0, 255, 0), width=2)

    ax, ay = to_px(pos)
    ar = 4
    d.ellipse((ax - ar, ay - ar, ax + ar, ay + ar), fill=(255, 0, 0))
    return img


def rollout_expert_gif(out_path: str, seed: int = 0):
    rng = np.random.RandomState(SEED + seed)
    pos = sample_pos(rng)
    goal = sample_pos(rng)

    frames = []
    for t in range(HORIZON):
        a = expert_action(pos, goal)
        pos = step(pos, a)

        img = render_pil(pos, goal).resize((IMG * 4, IMG * 4), Image.NEAREST)
        d = ImageDraw.Draw(img)
        d.text(
            (8, 8),
            f"EXPERT  t={t:02d}  dist={np.linalg.norm(goal - pos):.3f}",
            fill=(255, 255, 255),
        )
        frames.append(img)

        if np.linalg.norm(goal - pos) < GOAL_THRESH:
            break

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=60, loop=0
    )
    print(f"saved: {out_path}")


def rollout_policy_gif(encoder, policy, out_path: str, seed: int):
    rng = np.random.RandomState(SEED + seed)
    pos = sample_pos(rng)
    goal = sample_pos(rng)

    frames = []
    for t in range(HORIZON):
        img_np = np.asarray(render_pil(pos, goal), dtype=np.uint8)
        x = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = x.to(DEVICE)

        with torch.no_grad():
            z = encoder(x)
            a = policy(z).squeeze(0).cpu().numpy().astype(np.float32)

        a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
        pos = step(pos, a)

        img = render_pil(pos, goal).resize((IMG * 4, IMG * 4), Image.NEAREST)
        d = ImageDraw.Draw(img)
        d.text(
            (8, 8),
            f"POLICY  t={t:02d}  dist={np.linalg.norm(goal - pos):.3f}",
            fill=(255, 255, 255),
        )
        frames.append(img)

        if np.linalg.norm(goal - pos) < GOAL_THRESH:
            break

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=60, loop=0
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # for s in range(100):
    #     rollout_expert_gif(f"expert_rollout_{s:03d}.gif", seed=s)
    # exit(0)

    print("building dataset...")
    ds = ExpertDataset(N_SAMPLES, seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    print("loading DINO encoder...")
    encoder, emb_dim = load_dino_encoder(DINO_NAME, DEVICE)
    print(f"DINO embedding dim = {emb_dim}")

    policy = PolicyHead(emb_dim).to(DEVICE)

    print("training BC policy head...")
    train_bc(encoder, policy, dl)

    print("sanity rollout eval...")
    rollout_eval(encoder, policy)

    print("saving 20 random policy rollouts...")
    for i in range(20):
        rollout_policy_gif(
            encoder,
            policy,
            out_path=f"policy_rollout_{i:02d}.gif",
            seed=1000 + i,
        )


if __name__ == "__main__":
    main()
