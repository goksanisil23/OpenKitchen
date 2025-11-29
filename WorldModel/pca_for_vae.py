import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

LATENT_DATASET = "latent_dataset.npz"


def load_latent_npz(path):
    data = np.load(path, allow_pickle=False)
    z = data["z"].astype(np.float32)  # (N, dz)
    a = data["a"].astype(np.float32)  # (N, da=2)
    paths = data["paths"]  # (N,)
    paths = data["paths"].astype(str).tolist()
    return z, a, paths


z, a, img_paths = load_latent_npz(LATENT_DATASET)
N, z_dim = z.shape
print("loaded latent dataset: z:", z.shape, "a:", a.shape)
assert a.shape[0] == N, "Check action length"
assert len(img_paths) == N, "Check image paths length"
action_dim = a.shape[1]
print(f"Loaded {N} samples, z_dim={z_dim} a_dim={action_dim}")


# Randomly sample a subset for PCA fitting
D = z.shape[1]
z = torch.from_numpy(z)
print("Fitting PCA on latent vectors of dimension:", D)
z_mean = torch.mean(z, dim=0)
z_centered = z - z_mean
print(f"z_centered: {z_centered.shape}")

# SVD
U, S, V = torch.pca_lowrank(z_centered, q=D)

# Analyze the variance
explained_variance = (S**2) / (N - 1)
total_variance = explained_variance.sum()
cumulative_variance = torch.cumsum(explained_variance / total_variance, dim=0)

# find the cutoff for 95% of the information
k95 = (cumulative_variance >= 0.95).nonzero()[0].item() + 1
print(f"Number of PCA components to retain 95% variance: {k95} out of {D}")

# Plot
plt.plot(cumulative_variance.numpy())
plt.axhline(y=0.95, color="r", linestyle="--")
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative variance")
plt.title("PCA Cumulative Explained Variance")
plt.show()


class PCATransform(nn.Module):
    def __init__(self, mean_vector, v_matrix, k):
        super().__init__()
        self.register_buffer("mean", mean_vector)
        # Keep k columns of V
        # V is (128, 128), we want (128, k)
        self.projection = nn.Linear(128, k, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(v_matrix[:, :k].t())

    def forward(self, x):
        return self.projection(x - self.mean)


model = PCATransform(z_mean, V, k95)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("cpp_models/pca_transform.pt")
