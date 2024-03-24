import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SensorMeasurementsDataset(Dataset):
    def __init__(self, directory_path):
        self.measurements = []
        self.actions = []

        # Load and parse each file
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                lines = file.readlines()
                # Assuming the first 7 lines are sensor measurements and the last line is the action
                measurement = [
                    list(map(float, line.strip().split())) for line in lines[:-1]
                ]
                action = list(map(float, lines[-1].strip().split()))

                self.measurements.append(measurement)
                self.actions.append(action)

        # Convert lists to tensors
        self.measurements = torch.tensor(self.measurements, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)

    def __len__(self):
        return len(self.measurements)

    def __getitem__(self, idx):
        return self.measurements[idx], self.actions[idx]


# USAGE:

# data_directory_path = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/pointcloud_and_actions"
# dataset = SensorMeasurementsDataset(data_directory_path)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# OR
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Example: Print the shape of the loaded sensor measurements and action
# for measurement, action in dataloader:
#     print("measurement shape:", measurement.shape)  # Should be [batch_size, 7, 2]
#     print("action shape:", action.shape)  # Should be [batch_size, 2]
#     break
