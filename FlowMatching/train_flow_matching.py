import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from flow_matching_model import (
    ActionNormalizer,
    BEVActionDataset,
    ConditionalFlowMatchingPolicy,
)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"]
        x1 = batch["action_norm"]  # target action in normalized space

        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)

        # Straight-line path between x0 and x1
        x_t = (1.0 - t) * x0 + t * x1

        # CFM target velocity for straight path
        u_t = x1 - x0

        pred_u_t = model(x_t=x_t, t=t, image=image)
        loss = F.mse_loss(pred_u_t, u_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        image = batch["image"]
        x1 = batch["action_norm"]

        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)

        x_t = (1.0 - t) * x0 + t * x1
        u_t = x1 - x0

        pred_u_t = model(x_t=x_t, t=t, image=image)
        loss = F.mse_loss(pred_u_t, u_t)
        total_loss += loss.item() * batch_size

    return total_loss / len(loader.dataset)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train conditional flow matching policy on BEV-action data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing paired .png and .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Where to save model checkpoints",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--bev-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main():
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required. Training always preloads the full dataset to GPU."
        )

    normalizer = ActionNormalizer()
    print("Preloading all samples to GPU...")
    dataset = BEVActionDataset(
        data_dir=args.data_dir, image_size=args.image_size, normalizer=normalizer
    )

    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = ConditionalFlowMatchingPolicy(
        bev_dim=args.bev_dim, hidden_dim=args.hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    print(
        f"Training on {device} with {len(train_set)} train / {len(val_set)} val samples"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "image_size": args.image_size,
                "bev_dim": args.bev_dim,
                "hidden_dim": args.hidden_dim,
            },
            "normalizer": {
                "throttle_min": normalizer.throttle_min,
                "throttle_max": normalizer.throttle_max,
                "steering_min": normalizer.steering_min,
                "steering_max": normalizer.steering_max,
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(ckpt, output_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, output_dir / "best.pt")

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
