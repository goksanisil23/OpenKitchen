#!/usr/bin/env python3

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ScanDataset(Dataset):
    def __init__(self, folder_path):
        super(ScanDataset, self).__init__()
        self.data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as file:
                    points = []
                    for line in file:
                        x, y = map(float, line.split())
                        points.extend([x, y])
                    self.data.append(points)

        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(30, 20), nn.ReLU(), nn.Linear(20, 10)  # Encoded representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),  # Output size is 10 (5 2D points)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
