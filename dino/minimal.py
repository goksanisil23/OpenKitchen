# pip install timm torch torchvision matplotlib
import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load DINO ViT-S/16
model = timm.create_model("vit_small_patch16_224.dino", pretrained=True)
model.eval().to(device)

# Preprocess
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

IMAGE_DIR = Path(
    "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/test_dir"
)
image_paths = sorted(
    p
    for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)

if not image_paths:
    raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

plt.ion()
fig_pca, (ax_in, ax_tok) = plt.subplots(1, 2, figsize=(8, 4))
fig_cls, ax_cls = plt.subplots(1, 1, figsize=(10, 3))

for image_path in image_paths:
    # Load image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward features (gives patch tokens + CLS)
        print(f"input shape: {x.shape}")
        feats = model.forward_features(x)  # shape: [1, 197, 384] for ViT-S/16
        print("Feats shape:", feats.shape)
        cls_token = feats[:, 0, :]  # shape: [1, 384] -> latent vector
        patch_tokens = feats[:, 1:, :]  # shape: [1, 196, 384]

    # Example: reduce patch tokens to RGB for a quick visualization
    # (simple linear projection via PCA-like SVD on the tokens)
    pt = patch_tokens.squeeze(0)  # [196, 384]
    pt = pt - pt.mean(dim=0, keepdim=True)
    u, s, v = torch.pca_lowrank(pt, q=3)
    rgb = pt @ v[:, :3]  # [196, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    rgb = rgb.view(14, 14, 3).cpu().numpy()

    # Plot (reuse figures without closing)
    ax_in.clear()
    ax_tok.clear()
    ax_in.set_title(f"Input: {image_path.name}")
    ax_in.imshow(img.resize((224, 224)))
    ax_in.axis("off")
    ax_tok.set_title("Token PCA (14x14)")
    ax_tok.imshow(rgb)
    ax_tok.axis("off")
    fig_pca.tight_layout()
    fig_pca.canvas.draw()

    # CLS token bar plot (1 x 384)
    cls_vals = cls_token.squeeze(0).cpu().numpy()
    ax_cls.clear()
    ax_cls.set_title(f"CLS Token Bars: {image_path.name}")
    ax_cls.bar(np.arange(cls_vals.shape[0]), cls_vals, width=1.0)
    ax_cls.set_xlabel("Feature index")
    ax_cls.set_ylabel("Value")
    fig_cls.tight_layout()
    fig_cls.canvas.draw()
    plt.pause(1)

    print("Image:", image_path)
    print("Latent (CLS) shape:", cls_token.shape)
