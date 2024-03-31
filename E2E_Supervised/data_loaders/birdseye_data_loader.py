import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SensorMeasurementsDataset(Dataset):
    def __init__(
        self, directory_path, img_transform=None, grayscale=False, steer_only=False
    ):
        self.directory = directory_path
        self.img_transform = img_transform
        self.grayscale = grayscale
        self.steer_only = steer_only
        self.angle_normalization_limit = 15.0

        # Assume png and txt files have the same name
        self.filenames = [
            f[:-4] for f in os.listdir(directory_path) if f.endswith(".png")
        ]

        self._apply_data_based_angle_normalization()
        # self._filter_dataset()
        # self._filter_dataset_2()

    def _apply_data_based_angle_normalization(self):
        angles = []
        for f in self.filenames:
            action_path = os.path.join(self.directory, f + ".txt")
            with open(action_path, "r") as f:
                action = list(map(float, f.read().strip().split()))
                angles.append(action[1])

        min_val = min(angles)
        max_val = max(angles)
        if abs(min_val) > abs(max_val):
            self.angle_normalization_limit = abs(min_val)
        else:
            self.angle_normalization_limit = abs(max_val)

        print(f"Updated normalization to {self.angle_normalization_limit}")

    def _filter_dataset_2(self):
        filtered_filenames = []
        for filename in self.filenames:
            if "_IMS_" in filename:
                filtered_filenames.append(filename)
        self.filenames = filtered_filenames

    def _filter_dataset(self):
        ANGLE_LIMIT = 0.001
        STRAIGHT_RATIO = 5

        random.shuffle(self.filenames)

        # Portion of the dataset where robot drives relatively straight
        print(f"total files: {len(self.filenames)}")

        # Now filter based on actions with sufficient steering
        filtered_filenames = []
        straight_ctr = 1
        non_straight_ctr = 1
        for filename in self.filenames:
            action_path = os.path.join(self.directory, filename + ".txt")
            with open(action_path, "r") as f:
                action = list(map(float, f.read().strip().split()))
                if abs(action[1]) > ANGLE_LIMIT:
                    filtered_filenames.append(filename)
                    non_straight_ctr += 1
                elif (straight_ctr / non_straight_ctr) < (1 / STRAIGHT_RATIO):
                    filtered_filenames.append(filename)
                    straight_ctr += 1
        self.filenames = filtered_filenames
        print(f"Total data: {len(self.filenames)}")
        print(f"Straight: {straight_ctr}")
        print(f"Non-Straight: {non_straight_ctr}")

    def _normalize_actions(self, action):
        # Normalize to [-1,1]
        action[0] = (action[0] - 50) / 50.0  # normally [0,100]
        action[1] /= self.angle_normalization_limit
        return action

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.directory, self.filenames[idx] + ".png")
        action_path = os.path.join(self.directory, self.filenames[idx] + ".txt")

        if self.grayscale:
            image = Image.open(image_path).convert("L")
        else:
            image = Image.open(image_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)

        with open(action_path, "r") as f:
            action = list(map(float, f.read().strip().split()))
            action = self._normalize_actions(action)

            if self.steer_only:
                action = torch.tensor([action[1]], dtype=torch.float32)
            else:
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
