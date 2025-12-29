import argparse, os, random, json, math
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ----------------------------
# Dataset (flat folder, any class names not required)
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderFlat(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.paths = sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img  # (C,H,W) float in [0,1]


# ----------------------------
# Model
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, img_size=128, z_dim=64, ch=64):
        super().__init__()
        assert img_size % 16 == 0, "image_size must be divisible by 16"
        self.img_size = img_size
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.ReLU(True),  # s/2
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),  # s/4
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),  # s/8
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),  # s/16
        )
        h = img_size // 16
        self.flatten_dim = ch * 8 * h * h
        self.fc_mu = nn.Linear(self.flatten_dim, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)

    def forward(self, x):
        flat_feature_vec = self.net(x).view(x.size(0), -1)
        mu = self.fc_mu(flat_feature_vec)
        logvar = self.fc_logvar(flat_feature_vec)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, img_size=128, z_dim=64, ch=64):
        super().__init__()
        assert img_size % 16 == 0
        self.img_size = img_size
        self.z_dim = z_dim
        h = img_size // 16
        self.fc = nn.Linear(z_dim, ch * 8 * h * h)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),  # x2
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),  # x2
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),  # x2
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),  # x2 -> img_size
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, self.img_size // 16, self.img_size // 16)
        x = self.deconv(h)
        return x


class VAE(nn.Module):
    def __init__(self, img_size=128, z_dim=64, ch=64):
        super().__init__()
        self.enc = Encoder(img_size, z_dim, ch)
        self.dec = Decoder(img_size, z_dim, ch)

    # Reparametrization trick that allows to sample from a distribution
    # while keeping the gradient flow through the network.
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar


# ----------------------------
# Training / Eval
# ----------------------------
def vae_loss(recon, x, mu, logvar, beta=1.0):
    # ---- Reconstruction loss ----
    # BCE over pixels (expect inputs in [0,1])
    # Each pixel is treated as a Bernoulli variable, since the decoder outputs a probability per pixel via sigmoid
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum")
    # ---- Regularization loss ----
    # The term below is minimized at u=0, var=1
    # It forces the latent space to be a Gaussian
    # KL(N(mu, sigma) || N(0, I))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + beta * kld), recon_loss, kld


@torch.no_grad()
def visualize_in_grid(model, loader, device, out_path, n=16):
    model.eval()
    for batch in loader:
        x = batch.to(device)
        x = x[:n]
        recon, _, _ = model(x)
        # side-by-side: [x | recon] for each sample
        pairs = torch.cat([x, recon], dim=3)  # concat width
        save_image(pairs, out_path, nrow=int(math.sqrt(n)), padding=2)
        break


def build_loaders(args):
    tfm = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),  # [0,1]
        ]
    )
    if args.train_dir and args.test_dir:
        train_ds = ImageFolderFlat(args.train_dir, tfm)
        test_ds = ImageFolderFlat(args.test_dir, tfm)
    elif args.data_dir:
        full = ImageFolderFlat(args.data_dir, tfm)
        val_len = int(len(full) * args.val_ratio)
        train_len = len(full) - val_len
        train_ds, test_ds = random_split(
            full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )
    else:
        raise ValueError(
            "Provide --train_dir & --test_dir, or --data_dir (with --val_ratio)."
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # reproducibility-ish
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print("Loading training data...")
    train_loader, test_loader = build_loaders(args)

    model = VAE(img_size=args.image_size, z_dim=args.latent_dim, ch=args.base_ch).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        stats = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "n": 0}
        # for batch in train_loader:
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x = batch.to(device, non_blocking=True)
            recon, mu, logvar = model(x)
            loss, r, k = vae_loss(recon, x, mu, logvar, beta=args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            stats["loss"] += loss.item()
            stats["recon"] += r.item()
            stats["kld"] += k.item()
            stats["n"] += batch_size

        n = stats["n"]
        print(
            f"Epoch {epoch+1}/{args.epochs} | loss {(stats['loss']/n):.4f} | recon {(stats['recon']/n):.4f} | kld {(stats['kld']/n):.4f}"
        )

        # Save model after each epoch
        torch.save(model, os.path.join(args.output_dir, "vae.pt"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "vae_state.pt"))

    # Visual verification
    visualize_in_grid(
        model,
        train_loader,
        device,
        os.path.join(args.output_dir, "recon_train.png"),
        n=args.viz_n,
    )
    visualize_in_grid(
        model,
        test_loader,
        device,
        os.path.join(args.output_dir, "recon_test.png"),
        n=args.viz_n,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train a Conv VAE on robot camera frames")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--data_dir", type=str, help="Folder with all images (will split)")
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation/test split (if using --data_dir)",
    )

    p.add_argument(
        "--train_dir",
        type=str,
        default="../FieldNavigators/collect_data/build/measurements_and_actions/",
        help="Folder with training images",
    )
    p.add_argument(
        "--test_dir",
        type=str,
        default="../FieldNavigators/collect_data/build/test_dir/",
        help="Folder with test images",
    )

    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=1.0)

    p.add_argument(
        "--viz_n", type=int, default=16, help="How many samples to visualize per grid"
    )

    p.add_argument("--output_dir", type=str, default="vae_outputs")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
