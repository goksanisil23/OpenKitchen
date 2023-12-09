#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from laser_2d_autoencoder import Autoencoder, ScanDataset
from torch.utils.data import DataLoader

### HYPERPARAMETERS ###
N_EPOCH = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001

folder_path = "collect_data_racetrack/build/point_clouds/"
dataset = ScanDataset(folder_path)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training loop
for epoch in range(N_EPOCH):
    for data in data_loader:
        # Flatten the data: [batch_size, 5, 2] -> [batch_size, 10]
        input_data = data.view(data.size(0), -1).to(device)

        # Forward pass
        output = model(input_data)
        loss = criterion(output, input_data)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{N_EPOCH}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "laser_2d_autoencoder.pth")
