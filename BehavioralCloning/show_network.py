import numpy as np
import torch
import os

os.environ["MPLBACKEND"] = "Agg"  # or matplotlib.use("Agg") before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


import env_train_bc as train
from test_bc_env import obs_to_tensor, SIZE

IMAGE = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/SaoPaulo/birdseye_SAOPAULO_0_0.png"


def show_feats(t, title, k=16):
    # t: (1,C,H,W)
    f = t[0].detach().cpu().numpy()
    k = min(k, f.shape[0])
    n = int(np.ceil(np.sqrt(k)))
    plt.figure(figsize=(2.2 * n, 2.2 * n))
    for i in range(k):
        ax = plt.subplot(n, n, i + 1)
        ax.imshow(f[i], cmap="viridis")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"feature_maps_{title.replace(' ','_')}.png")


device = torch.device("cpu")

net = train.PolicyCNN().to(device)
img = Image.open(IMAGE).convert("RGB").resize(SIZE)
np_img = np.array(img, dtype=np.uint8)  # (96,96,3)
x = obs_to_tensor(np_img, device=device)

# x is your input tensor (1,3,96,96) in [0,1]
net.eval()
with torch.no_grad():
    feat1 = net.encoder[:2](x)  # conv1 + relu -> (1,32,H,W)
    feat2 = net.encoder[:4](x)  # conv2 + relu -> (1,64,H,W)
    feat3 = net.encoder[:6](x)  # conv3 + relu -> (1,64,H,W)

show_feats(feat1, "After conv1+ReLU", k=16)
show_feats(feat2, "After conv2+ReLU", k=16)
show_feats(feat3, "After conv3+ReLU", k=16)
