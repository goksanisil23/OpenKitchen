import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SensorMeasurementsDataset(Dataset):
    ANGLE_LIMIT = 0.25
    STRAIGHT_RATIO = 5

    def __init__(self, directory_path, img_transform=None):
        self.directory = directory_path
        self.img_transform = img_transform

        # Assume png and txt files have the same name
        self.filenames = [
            f[:-4] for f in os.listdir(directory_path) if f.endswith(".png")
        ]
        random.shuffle(self.filenames)

        # Portion of the dataset where robot drives relatively straight
        NUM_STRAIGHT_DATA = len(self.filenames) // self.STRAIGHT_RATIO
        print(f"total files: {len(self.filenames)}")
        print(f"NUM_STRAIGHT_DATA: {NUM_STRAIGHT_DATA}")

        # Now filter based on actions with sufficient steering
        filtered_filenames = []
        straight_ctr = 0
        for filename in self.filenames:
            action_path = os.path.join(self.directory, filename + ".txt")
            with open(action_path, "r") as f:
                action = list(map(float, f.read().strip().split()))
                if abs(action[1]) > self.ANGLE_LIMIT:
                    filtered_filenames.append(filename)
                elif straight_ctr < NUM_STRAIGHT_DATA:
                    filtered_filenames.append(filename)
                    straight_ctr += 1
        self.filenames = filtered_filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.filenames[idx] + ".png")
        action_path = os.path.join(self.directory, self.filenames[idx] + ".txt")

        image = Image.open(image_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)

        with open(action_path, "r") as f:
            action = list(map(float, f.read().strip().split()))
            # Normalize to [-1,1]
            assert action[0] <= 100.0
            assert abs(action[1]) <= 180.0
            action[0] = (action[0] - 50) / 50.0  # normally [0,100]
            action[1] /= 180.0  # normally [-180,180]
            action = torch.tensor(action, dtype=torch.float32)

        return image, action


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
