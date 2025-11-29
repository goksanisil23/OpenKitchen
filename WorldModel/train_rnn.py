# ---- train_gru_dynamics.py ----
import os, json, math, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image

from train_vae import VAE


# ----------------------------
# Dataset: sequences from (z, a, paths)
# ----------------------------
class LatentActionSeq(Dataset):
    def __init__(self, z, a, img_paths, starts, seq_len, z_mean, z_std, a_mean, a_std):
        self.z = z
        self.a = a
        self.img_paths = img_paths
        self.starts = starts
        self.seq_len = seq_len
        self.z_mean, self.z_std = z_mean, z_std
        self.a_mean, self.a_std = a_mean, a_std

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.seq_len
        # inputs: (z_t, a_t) for t in [start, end-1]
        z_t = (self.z[start:end] - self.z_mean) / self.z_std
        a_t = (self.a[start:end] - self.a_mean) / self.a_std
        x = np.concatenate([z_t, a_t], axis=-1)  # (T, dz+da)
        # targets: z_{t+1} for t in [start, end-1]
        y = (self.z[start + 1 : end + 1] - self.z_mean) / self.z_std  # (T, dz)
        # store gt (t+1) image img_paths for viz
        img_paths_next = self.img_paths[start + 1 : end + 1]
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y, img_paths_next


def make_starts(img_paths, seq_len):
    # sequences start at s and go up to s+seq_len; target uses +1
    # Do not use the samples at track boundaries

    def _parse_seq_id(img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        parts = base.split("_")
        idx = int(parts[-1])
        return idx

    seq_ids = []
    for p in img_paths:
        seq_id = _parse_seq_id(p)
        seq_ids.append(seq_id)

    N = len(img_paths)
    max_start = (N - 1) - seq_len
    assert max_start > 0, "Not enough data for even one sequence"

    valid_mask = [
        (
            seq_ids[k + seq_len - 1] == (seq_ids[k] + (seq_len - 1) * 10)
            if k + seq_len < len(seq_ids)
            else False
        )
        for k in range(len(seq_ids))
    ]

    valid_mask = valid_mask[: max_start + 1]

    start_ids = np.arange(0, max_start + 1, dtype=np.int64)
    return start_ids[valid_mask]


# ----------------------------
# Model
# ----------------------------
class GRUDynamics(nn.Module):
    def __init__(self, z_dim, a_dim, hidden=256, layers=3, dropout=0.0):
        super().__init__()
        self.input_dim = z_dim + a_dim
        self.hidden = hidden
        self.gru = nn.GRU(
            self.input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, z_dim)

    def forward(self, x):
        # x: (B, T, z+a)
        y, h = self.gru(x)  # y: (B,T,H)
        z_next = self.head(y)  # (B,T,dz)
        return z_next, h


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def load_latent_npz(path):
    data = np.load(path, allow_pickle=False)
    z = data["z"].astype(np.float32)  # (N, dz)
    a = data["a"].astype(np.float32)  # (N, da=2)
    paths = data["paths"]  # (N,)
    paths = data["paths"].astype(str).tolist()
    return z, a, paths


def compute_norm_stats(z_train, a_train, eps=1e-6):
    z_mean = z_train.mean(0, keepdims=True)
    z_std = z_train.std(0, keepdims=True) + eps
    a_mean = a_train.mean(0, keepdims=True)
    a_std = a_train.std(0, keepdims=True) + eps
    return z_mean, z_std, a_mean, a_std


# ----------------------------
# Train
# ----------------------------
def train(args):
    set_seed(42)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    z, a, img_paths = load_latent_npz(args.npz_path)
    N, z_dim = z.shape
    print("loaded latent dataset: z:", z.shape, "a:", a.shape)
    assert a.shape[0] == N, "Check action length"
    assert len(img_paths) == N, "Check image paths length"
    action_dim = a.shape[1]
    print(f"Loaded {N} samples from {args.npz_path}, z_dim={z_dim} a_dim={action_dim}")

    # indices of sequence starts (sliding window)
    starts = make_starts(img_paths, args.seq_len)
    # for s in starts:
    #     print(f"valid img start: {img_paths[s]}")
    # split on sequence-start level
    num_seq = len(starts)
    val_len = int(num_seq * args.val_ratio)
    train_len = num_seq - val_len
    idx = np.arange(num_seq)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    train_ids = np.sort(idx[:train_len])
    val_ids = np.sort(idx[train_len:])

    train_starts = starts[train_ids]
    val_starts = starts[val_ids]

    # normalize on training windowed data (use samples that appear in train sequences only)
    mask = np.zeros(N, dtype=bool)
    for s in train_starts:
        mask[s : s + args.seq_len] = True
    print(f"Using {mask.sum()}/{N} samples for computing normalization stats")
    z_mean, z_std, a_mean, a_std = compute_norm_stats(z[mask], a[mask])
    # save stats
    np.savez_compressed(
        os.path.join(args.output_dir, "stats.npz"),
        z_mean=z_mean,
        z_std=z_std,
        a_mean=a_mean,
        a_std=a_std,
    )

    train_ds = LatentActionSeq(
        z, a, img_paths, train_starts, args.seq_len, z_mean, z_std, a_mean, a_std
    )
    val_ds = LatentActionSeq(
        z, a, img_paths, val_starts, args.seq_len, z_mean, z_std, a_mean, a_std
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model/opt
    model = GRUDynamics(
        z_dim=z_dim,
        a_dim=action_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # save config
    cfg = vars(args).copy()
    cfg.update({"z_dim": z_dim, "N": int(N)})
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        t_loss = 0.0
        t_n = 0
        for x, y, _ in train_loader:
            x = x.to(device)  # (B,T,dz+da) normalized
            y = y.to(device)  # (B,T,dz)    normalized
            z_pred, _ = model(x)  # teacher forcing: always feed GT (z_t, a_t)
            loss = F.mse_loss(z_pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()
            t = x.size(0) * x.size(1)
            t_loss += loss.item() * t
            t_n += t
        sched.step()
        train_loss = t_loss / max(1, t_n)

        # ---- val ----
        model.eval()
        v_loss = 0.0
        v_n = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                z_pred, _ = model(x)
                loss = F.mse_loss(z_pred, y)
                t = x.size(0) * x.size(1)
                v_loss += loss.item() * t
                v_n += t
        val_loss = v_loss / max(1, v_n)

        print(
            f"Epoch {epoch}/{args.epochs} | train {train_loss:.6f} | val {val_loss:.6f}"
        )

        # checkpoint
        ck = {
            "model": model.state_dict(),
            "args": cfg,
        }
        torch.save(ck, os.path.join(args.output_dir, "gru_dyn_latest.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ck, os.path.join(args.output_dir, "gru_dyn_best.pt"))


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser("Train GRU dynamics: (z_t, a_t) -> z_{t+1}")
    p.add_argument("--npz_path", type=str, default="latent_dataset.npz")
    p.add_argument("--vae_ckpt", type=str, default="vae_outputs/vae_state.pt")
    p.add_argument("--output_dir", type=str, default="dyn_outputs")
    p.add_argument("--seq_len", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
