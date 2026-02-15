import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ActionNormalizer:
    """Normalizes actions to approximately [-1, 1]."""

    throttle_min: float = 0.0
    throttle_max: float = 100.0
    steering_min: float = -10.0
    steering_max: float = 10.0

    def normalize(self, throttle: float, steering: float) -> Tuple[float, float]:
        throttle_norm = (
            2.0
            * (throttle - self.throttle_min)
            / (self.throttle_max - self.throttle_min)
            - 1.0
        )
        steering_center = 0.5 * (self.steering_max + self.steering_min)
        steering_half_range = 0.5 * (self.steering_max - self.steering_min)
        steering_norm = (steering - steering_center) / steering_half_range
        return throttle_norm, steering_norm

    def denormalize(
        self, throttle_norm: float, steering_norm: float
    ) -> Tuple[float, float]:
        throttle = (0.5 * (throttle_norm + 1.0)) * (
            self.throttle_max - self.throttle_min
        ) + self.throttle_min
        steering_center = 0.5 * (self.steering_max + self.steering_min)
        steering_half_range = 0.5 * (self.steering_max - self.steering_min)
        steering = steering_norm * steering_half_range + steering_center
        return throttle, steering


def parse_action_file(path: Path) -> Tuple[float, float]:
    """
    Parses one-line action file and extracts throttle + steering values.
    Accepts plain values like: "42.1 -3.0" or text containing two numbers.
    """
    text = path.read_text().strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numbers) < 2:
        raise ValueError(f"Could not parse two action values in: {path}")
    throttle = float(numbers[0])
    steering = float(numbers[1])
    return throttle, steering


def preprocess_bev_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    # HWC -> CHW
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class BEVActionDataset(Dataset):
    """
    Pairs BEV png/txt files and preloads all processed tensors once to GPU.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 128,
        normalizer: ActionNormalizer | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("BEVActionDataset requires CUDA to preload data to GPU.")
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.normalizer = normalizer or ActionNormalizer()
        self.samples = self._build_index()
        self.preload_device = torch.device("cuda")
        self._preload()

    def _build_index(self) -> List[Tuple[Path, Path]]:
        txt_files = sorted(self.data_dir.glob("*.txt"))
        samples = []
        for txt_path in txt_files:
            png_path = txt_path.with_suffix(".png")
            if png_path.exists():
                samples.append((png_path, txt_path))

        if not samples:
            raise RuntimeError(
                f"No (png, txt) pairs found in {self.data_dir}. "
                "Expected files such as birdseye_SAOPAULO_993.png and birdseye_SAOPAULO_993.txt"
            )
        return samples

    def _preload(self):
        images = []
        actions = []
        image_paths = []

        for image_path, action_path in tqdm(self.samples, desc="Preloading"):
            image = preprocess_bev_image(image_path, self.image_size)
            throttle, steering = parse_action_file(action_path)
            throttle_norm, steering_norm = self.normalizer.normalize(throttle, steering)
            action_norm = torch.tensor(
                [throttle_norm, steering_norm], dtype=torch.float32
            )
            images.append(image)
            actions.append(action_norm)
            image_paths.append(str(image_path))

        self.images = torch.stack(images, dim=0).to(self.preload_device)
        self.actions = torch.stack(actions, dim=0).to(self.preload_device)
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return {
            "image": self.images[idx],
            "action_norm": self.actions[idx],
            "image_path": self.image_paths[idx],
        }


class BEVEncoder(nn.Module):
    """Encodes a BEV image into a compact feature vector."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.conv(image)  # [B, 128, 1, 1]
        features = features.flatten(1)  # [B, 128]
        return self.proj(features)


class ActionFlowTrunk(nn.Module):
    """
    Predicts flow velocity u_t for action vector given:
    - noisy/intermediate action x_t
    - time t
    - BEV condition embedding
    """

    def __init__(self, bev_dim: int = 128, hidden_dim: int = 256, action_dim: int = 2):
        super().__init__()
        input_dim = action_dim + 1 + bev_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, bev_embedding: torch.Tensor
    ) -> torch.Tensor:
        model_input = torch.cat([x_t, t, bev_embedding], dim=1)
        return self.net(model_input)


class ConditionalFlowMatchingPolicy(nn.Module):
    """Combines BEV encoder and flow trunk for conditional flow matching."""

    def __init__(self, bev_dim: int = 128, hidden_dim: int = 256, action_dim: int = 2):
        super().__init__()
        self.bev_encoder = BEVEncoder(embed_dim=bev_dim)
        self.action_flow_trunk = ActionFlowTrunk(
            bev_dim=bev_dim, hidden_dim=hidden_dim, action_dim=action_dim
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, image: torch.Tensor
    ) -> torch.Tensor:
        bev_embedding = self.bev_encoder(image)
        flow_velocity = self.action_flow_trunk(x_t, t, bev_embedding)
        return flow_velocity
