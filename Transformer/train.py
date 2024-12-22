import glob
import os

import numpy as np
import torch
import torch.nn as nn
from laser_transformer import (
    LidarTransformer,
    denormalize_controls,
    normalize_controls,
    normalize_laser_point,
)
from torch.utils.data import DataLoader, Dataset


class LidarControlDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Directory containing the timestep txt files
        """
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Read the text file
        file_path = self.file_paths[idx]
        data = np.loadtxt(file_path)

        # First 7 rows are laser scan points
        lidar_points = data[:7, :]  # Shape: (7, 2)

        # Last row contains controls [throttle, steering]
        controls = data[-1, :]  # Shape: (2,)

        # Normalize data
        lidar_points = np.array(
            [normalize_laser_point(point) for point in lidar_points]
        )
        controls = normalize_controls(controls)

        return torch.FloatTensor(lidar_points), torch.FloatTensor(controls)


def prepare_dataloaders(data_dir, train_split=0.8, batch_size=32):
    """
    Prepare train and validation dataloaders
    """
    # Get all file paths
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))

    # Split into train and validation
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Create train and validation directories
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create symbolic links to original files
    for file_path in train_files:
        target_path = os.path.join(train_dir, os.path.basename(file_path))
        if not os.path.exists(target_path):
            os.symlink(file_path, target_path)
    for file_path in val_files:
        target_path = os.path.join(val_dir, os.path.basename(file_path))
        if not os.path.exists(target_path):
            os.symlink(file_path, target_path)

    # Create datasets and dataloaders
    train_dataset = LidarControlDataset(train_dir)
    val_dataset = LidarControlDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for lidar_scans, controls in train_loader:
            lidar_scans = lidar_scans.to(device)
            controls = controls.to(device)

            optimizer.zero_grad()
            predictions = model(lidar_scans)
            loss = criterion(predictions, controls)
            loss.backward()
            optimizer.step()

            # Calculate unnormalized MSE for interpretability
            with torch.no_grad():
                unnorm_pred = torch.tensor(
                    denormalize_controls(predictions.cpu().numpy())
                )
                unnorm_true = torch.tensor(denormalize_controls(controls.cpu().numpy()))
                unnorm_loss = nn.MSELoss()(unnorm_pred, unnorm_true)

            train_loss += unnorm_loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lidar_scans, controls in val_loader:
                lidar_scans = lidar_scans.to(device)
                controls = controls.to(device)

                predictions = model(lidar_scans)

                # Calculate unnormalized MSE
                unnorm_pred = torch.tensor(
                    denormalize_controls(predictions.cpu().numpy())
                )
                unnorm_true = torch.tensor(denormalize_controls(controls.cpu().numpy()))
                unnorm_loss = nn.MSELoss()(unnorm_pred, unnorm_true)
                val_loss += unnorm_loss.item()

        val_loss /= len(val_loader)

        # # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                "best_model.pth",
            )

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss (unnormalized): {train_loss:.6f}")
        print(f"Validation Loss (unnormalized): {val_loss:.6f}")


#################################################
data_dir = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions_IMS"
train_loader, val_loader = prepare_dataloaders(data_dir, train_split=0.8, batch_size=32)
model = LidarTransformer(n_points=7)
train_model(model, train_loader, val_loader, num_epochs=10)
