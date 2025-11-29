# test_rnn.py
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from inspect import signature
import cv2


from train_vae import VAE

from train_rnn import (
    GRUDynamics,
    LatentActionSeq,
    make_starts,
    load_latent_npz,
)


def load_vae(vae_ckpt_path, device):
    vae_dir = Path(vae_ckpt_path).parent
    cfg = json.load(open(vae_dir / "args.json"))
    state = torch.load(vae_ckpt_path, map_location="cpu")
    vae = VAE(
        img_size=cfg["image_size"], z_dim=cfg["latent_dim"], ch=cfg["base_ch"]
    ).to(device)
    vae.load_state_dict(state)
    vae.eval()
    return vae, cfg


# torch -> numpy (RGB 0..1) -> BGR uint8 for OpenCV
def to_bgr_uint8(t_img):
    img = t_img.detach().cpu().permute(1, 2, 0).numpy()  # H,W,3 RGB
    img = (img * 255.0).round().astype(np.uint8)
    return img[:, :, ::-1]  # RGB->BGR


@torch.no_grad()
def visualize_stepwise(
    image_size, rnn_model, vae_model, val_loader, device, z_mean, z_std
):
    rnn_model.eval()
    vae_model.eval()

    z_mean = z_mean.to(device)
    z_std = z_std.to(device)

    cv2.namedWindow("RNN Validation", cv2.WINDOW_NORMAL)

    for x, y, paths_next in val_loader:
        x = x.to(device)  # (B,T,dz+da) --> input to RNN
        y = y.to(device)  # (B,T,dz) --> GT latent vector = z_t+1
        print(f"RNN input shape: {x.shape}")
        zhat_norm, _ = rnn_model(x)  # (B,T,dz)

        B, T, dz = zhat_norm.shape

        for b in range(B):
            for t in range(T):
                # denorm + decode
                z_pred = zhat_norm[b, t] * z_std + z_mean  # (dz,)
                z_gt = y[b, t] * z_std + z_mean  # (dz,)

                pred_img = vae_model.decode(z_pred.unsqueeze(0)).squeeze(0).clamp(0, 1)
                gt_img = vae_model.decode(z_gt.unsqueeze(0)).squeeze(0).clamp(0, 1)

                pred_np = to_bgr_uint8(pred_img)
                gt_np = to_bgr_uint8(gt_img)

                # raw image via OpenCV (already BGR)
                p = paths_next[t][b]  # time-major [T][B]
                raw = cv2.imread(p, cv2.IMREAD_COLOR)
                raw = cv2.resize(
                    raw, (image_size, image_size), interpolation=cv2.INTER_LINEAR
                )

                triplet = np.concatenate([raw, gt_np, pred_np], axis=1)
                caption = f"val, [raw | decode(z_gt) | decode(z_pred)]"
                vis = triplet.copy()
                cv2.putText(
                    vis,
                    caption,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("RNN Validation", vis)
                k = cv2.waitKey(0) & 0xFF  # wait for key per image
                if k in (ord("q"), 27):  # 'q' or ESC to quit
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()


def build_model(z_dim, a_dim, cfg, device):
    hidden = int(cfg.get("hidden", 256))
    layers = int(cfg.get("layers", 1))
    dropout = float(cfg.get("dropout", 0.0))
    kwargs = dict(
        z_dim=z_dim, a_dim=a_dim, hidden=hidden, layers=layers, dropout=dropout
    )
    # pass optional flags only if the class supports them (e.g., residual)
    params = signature(GRUDynamics.__init__).parameters
    for opt_key in ("residual",):
        if opt_key in params and opt_key in cfg:
            kwargs[opt_key] = cfg[opt_key]
    model = GRUDynamics(**kwargs).to(device)
    return model


def main():
    p = argparse.ArgumentParser("Load trained RNN and visualize predictions (OpenCV)")
    p.add_argument("--npz_path", type=str, default="latent_dataset.npz")
    p.add_argument("--rnn_ckpt", type=str, default="dyn_outputs/gru_dyn_best.pt")
    p.add_argument("--vae_ckpt", type=str, default="vae_outputs/vae_state.pt")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # --- load checkpoint + its cfg ---
    ck = torch.load(args.rnn_ckpt, map_location="cpu")
    cfg = ck.get("args", {})
    seq_len = int(cfg.get("seq_len", 5))
    seq_len = 1

    # --- data + stats ---
    z, a, img_paths = load_latent_npz(args.npz_path)
    _, z_dim = z.shape
    a_dim = a.shape[1]

    stats_path = Path(args.rnn_ckpt).parent / "stats.npz"
    stats = np.load(stats_path)
    z_mean = stats["z_mean"].astype(np.float32)
    z_std = stats["z_std"].astype(np.float32)
    a_mean = stats["a_mean"].astype(np.float32)
    a_std = stats["a_std"].astype(np.float32)

    starts = make_starts(img_paths, seq_len)
    dataset = LatentActionSeq(
        z, a, img_paths, starts, seq_len, z_mean, z_std, a_mean, a_std
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # --- model ---
    rnn_model = build_model(z_dim, a_dim, cfg, device)
    rnn_model.load_state_dict(ck["model"], strict=True)
    rnn_model.eval()
    print(f"rnn config: {cfg}")

    # --- vae ---
    vae_model, vae_cfg = load_vae(args.vae_ckpt, device)
    image_size = int(vae_cfg["image_size"])

    # --- visualize ---
    z_mean_t = torch.from_numpy(z_mean).to(device)
    z_std_t = torch.from_numpy(z_std).to(device)
    visualize_stepwise(
        image_size=image_size,
        rnn_model=rnn_model,
        vae_model=vae_model,
        val_loader=val_loader,
        device=device,
        z_mean=z_mean_t,
        z_std=z_std_t,
    )


if __name__ == "__main__":
    main()
