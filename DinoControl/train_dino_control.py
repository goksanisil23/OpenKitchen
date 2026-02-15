import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import numpy as np

# ==========================================
# 1. HARDCODED PARAMETERS
# ==========================================
DATA_DIR = (
    "/home/goksan/OpenKitchen/FieldNavigators/collect_data/build/SaoPaulo_random/"
)

# MODEL_NAME = "vit_small_patch16_224.dino"
MODEL_NAME = "vit_small_patch8_224.dino"
# MODEL_NAME = "openvision-vit-tiny-patch8-224"
BATCH_SIZE = 512  # 256
LEARNING_RATE = 1e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_IMG_SIZE = 224

VAL_SPLIT = 0.2
SPLIT_SEED = 31


TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ==========================================
# 2. DATASET
# ==========================================
class DrivingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform

        self.images = []
        self.labels = []

        # Find all png files
        png_files = glob.glob(os.path.join(root_dir, "*.png"))

        for png_path in tqdm(png_files):
            txt_path = png_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                img = (
                    Image.open(png_path)
                    .convert("RGB")
                    .resize((DINO_IMG_SIZE, DINO_IMG_SIZE), Image.NEAREST)
                )
                img.load()
                self.images.append(img)
                with open(txt_path) as f:
                    vals = f.read().strip().split()
                    throttle = float(vals[0]) / 100.0  # Normalize to [0, 1]
                    steering = float(vals[1]) / 10.0  # Normalize to [-1, 1]
                    # self.labels.append(
                    #     torch.tensor([throttle, steering], dtype=torch.float32)
                    # )
                    # Just use steering
                    self.labels.append(torch.tensor([steering], dtype=torch.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Transform is delayed to training phase since it converts uint8 to float32
        if self.transform:
            img = self.transform(img)

        return img, label


# ==========================================
# 3. MODEL
# ==========================================
class PolicyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(64, 1)

    def forward(self, z):
        features = self.net(z)
        raw_out = self.output_layer(features)
        return raw_out


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.num_features
        self.head = PolicyHead(self.embed_dim)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def weighted_mse(outputs, labels, device):
    weights = torch.ones_like(labels).to(device)
    weights[labels.abs() > 0.1] = 5.0  # Higher weight for larger steering angles
    return (weights * (outputs - labels) ** 2).mean()


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    print(f"Training on device: {DEVICE}")

    # Initialize Datasets
    dataset = DrivingDataset(DATA_DIR, transform=TRANSFORM)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Found {train_size} training samples and {val_size} test samples.")

    model = Agent().to(DEVICE)
    optimizer = optim.Adam(model.head.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            loss = weighted_mse(outputs, labels, DEVICE)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                # loss = criterion(outputs, labels)
                loss = weighted_mse(outputs, labels, DEVICE)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(val_loader)

        print(
            f"\nEpoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}"
        )
        torch.save(model.state_dict(), f"agent_model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
