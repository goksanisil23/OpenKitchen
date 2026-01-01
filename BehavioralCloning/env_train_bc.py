import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2


class CarRacingBCDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.rgb = data["rgb"]  # uint8, (N,96,96,3)
        self.act = data["act"]  # float32, (N,1)

        # SHow a histogram of steering values
        # steer_values = self.act[:, 0]

        # cv2.imshow("rgb_sample", cv2.cvtColor(self.rgb[0], cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)

    def __len__(self):
        return self.rgb.shape[0]

    def __getitem__(self, idx):
        x = self.rgb[idx].astype(np.float32) / 255.0  # (96,96,3)
        x = np.transpose(x, (2, 0, 1))  # (3,96,96)
        y = self.act[idx].astype(np.float32)  # (3,)
        return torch.from_numpy(x), torch.from_numpy(y)


class PolicyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # infer flatten dim with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            enc_dim = self.encoder(dummy).shape[-1]

        # Note: Convolutions are translation invariant. But this final mapping (steering head) is not
        # translation invariant (by design).
        # Because flatten + fully connected layers assign different weights to different spatial positions,
        # which is desired for steering control.
        self.steering_head = nn.Sequential(
            nn.Linear(enc_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        steer = self.steering_head(z)

        # Enforce action bounds: steer in [-1,1]
        return torch.tanh(steer)


def main(
    data_path="sao_paulo_dataset_mixed.npz",
    out_path="sao_paulo_policy.pt",
    batch_size=256,
    lr=3e-4,
    epochs=300,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    ds = CarRacingBCDataset(data_path)
    print(f"Dataset size: {len(ds)} samples")
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    print("Starting training...")

    net = PolicyCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        net.train()
        running = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            pred = net(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()

        print(f"epoch {ep:02d} | loss {running / len(dl):.6f}")

    torch.save(net.state_dict(), out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
