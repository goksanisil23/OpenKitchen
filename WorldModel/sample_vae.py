# sample_vae.py
import argparse, math
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from WorldModel.train_vae import VAE
import json


def slerp(a, b, t):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    dot = (a_norm * b_norm).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


def load_img(path, size):
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)  # [1,3,H,W]


@torch.no_grad()
def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model_config = json.load(open(Path(args.model).with_name("args.json")))
    state_dict = torch.load(args.model, map_location="cpu")
    model = VAE(
        img_size=model_config["image_size"],
        z_dim=model_config["latent_dim"],
        ch=model_config["base_ch"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    img_size = getattr(model.dec, "img_size", args.image_size)
    z_dim = model.dec.fc.in_features

    # 1) Pure sampling: z ~ N(0, tau^2 I)
    torch.manual_seed(args.seed)
    n = args.num_samples
    z = torch.randn(n, z_dim, device=device) * args.tau
    samples = model.decode(z).clamp(0, 1)
    save_image(samples, Path(args.outdir) / "samples.png", nrow=int(math.sqrt(n)))

    # 2) Latent interpolation between two real images
    interp_img_1 = "../FieldNavigators/collect_data/build/train_dir/measurements_and_actions/birdseye_ZANDVOORT_2_1330.png"
    interp_img_2 = "../FieldNavigators/collect_data/build/train_dir/measurements_and_actions/birdseye_ZANDVOORT_2_1400.png"
    xa = load_img(interp_img_1, img_size).to(device)
    xb = load_img(interp_img_2, img_size).to(device)
    za, mu_a, logvar_a = model.encode(xa)
    zb, mu_b, logvar_b = model.encode(xb)
    # use means for clean path
    za, zb = mu_a, mu_b
    steps = args.interp_steps
    grid = []
    for i in range(steps):
        t = torch.tensor(i / (steps - 1), device=device)
        zt = slerp(za, zb, t)
        xt = model.decode(zt).clamp(0, 1)
        grid.append(xt)
    grid = torch.cat(grid, dim=0)
    save_image(grid, Path(args.outdir) / "interpolation.png", nrow=steps)

    # 3) Stochastic recon: same image, multiple samples q(z|x)
    stochastic = "../FieldNavigators/collect_data/build/train_dir/measurements_and_actions/birdseye_ZANDVOORT_2_1330.png"
    x = load_img(stochastic, img_size).to(device)
    z, mu, logvar = model.encode(x)
    k = args.k
    outs = []
    for _ in range(k):
        # reparameterize with fresh noise
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar) * args.tau
        outs.append(model.decode(z).clamp(0, 1))
    outs = torch.cat(outs, dim=0)
    # prepend the original on the first row
    save_image(
        torch.cat([x.clamp(0, 1), outs], dim=0),
        Path(args.outdir) / "stochastic_recon.png",
        nrow=min(k + 1, 9),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="outputs/vae_state.pt")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--interp_steps", type=int, default=8)

    p.add_argument("--k", type=int, default=8)
    p.add_argument(
        "--tau", type=float, default=1.0, help="sampling temperature (std scale)"
    )
    p.add_argument(
        "--image_size", type=int, default=128
    )  # fallback if model lacks attr
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
