from pathlib import Path
import numpy as np
import cv2

import torch
import timm

import matplotlib

OUTPUT_PATH = Path("dino_vits16_ts.pt")

output_path = OUTPUT_PATH.expanduser().resolve()
print(f"Exporting TorchScript model to: {output_path}")

# Use the scriptable variant so we can script (keeps methods like forward_features).
model = timm.create_model(
    "vit_small_patch16_224.dino", pretrained=True, scriptable=True
)
model.eval()

with torch.no_grad():
    scripted = torch.jit.script(model)
scripted.save(output_path.as_posix())
print("Done.")

####### Visualize ########
img = cv2.imread(
    "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/test_dir/birdseye_ZANDVOORT_0.png",
    cv2.IMREAD_COLOR,
)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
img = img.astype(np.float32) / 255.0
x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
x = (x - 0.5) / 0.5


with torch.no_grad():
    out = scripted.forward_features(x)
    print("Reference output shape:", out.shape)
    cls_token = out[:, 0, :]
    patch_tokens = out[:, 1:, :]

# Visualization
matplotlib.use("Agg")  # headless-friendly backend to avoid Qt plugin issues
import matplotlib.pyplot as plt

pt = patch_tokens.squeeze(0)
pt = pt - pt.mean(dim=0, keepdim=True)
_, _, v = torch.pca_lowrank(pt, q=3)
rgb = pt @ v[:, :3]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
rgb = rgb.view(14, 14, 3).cpu().numpy()

# Plot input + token PCA
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title(f"Input")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Token PCA (14x14)")
plt.imshow(rgb)
plt.axis("off")
plt.tight_layout()
plt.savefig("dino_vits16_visualization.png")
# plt.show()

# Plot CLS token bar chart
cls_vals = cls_token.squeeze(0).cpu().numpy()
plt.figure(figsize=(10, 3))
plt.title(f"CLS Token Bars")
plt.bar(np.arange(cls_vals.shape[0]), cls_vals, width=1.0)
plt.xlabel("Feature index")
plt.ylabel("Value")
plt.tight_layout()
# plt.show()
# plt.savefig("dino_vits16_visualization.png")
