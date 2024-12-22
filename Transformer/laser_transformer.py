import numpy as np
import torch
import torch.nn as nn

# Normalization constants
LASER_X_RANGE = (-200.0, 200.0)  # Adjust these values based on your laser scanner specs
LASER_Y_RANGE = (-200.0, 200.0)  # Adjust these values based on your laser scanner specs
THROTTLE_RANGE = (0, 100.0)  # Typical range for throttle control
STEERING_RANGE = (-2.0, 2.0)  # Typical range for steering control, degrees


def normalize_laser_point(point):
    """Normalize laser point coordinates to [-1, 1] range"""
    x, y = point
    x_norm = 2 * (x - LASER_X_RANGE[0]) / (LASER_X_RANGE[1] - LASER_X_RANGE[0]) - 1
    y_norm = 2 * (y - LASER_Y_RANGE[0]) / (LASER_Y_RANGE[1] - LASER_Y_RANGE[0]) - 1
    return np.array([x_norm, y_norm])


def normalize_controls(controls):
    """Normalize throttle and steering to [-1, 1] range"""
    throttle, steering = controls
    throttle_norm = (
        2 * (throttle - THROTTLE_RANGE[0]) / (THROTTLE_RANGE[1] - THROTTLE_RANGE[0]) - 1
    )
    steering_norm = (
        2 * (steering - STEERING_RANGE[0]) / (STEERING_RANGE[1] - STEERING_RANGE[0]) - 1
    )
    return np.array([throttle_norm, steering_norm])


def denormalize_controls(normalized_controls):
    """Convert normalized controls back to original range. Works with both single predictions and batches."""
    normalized_controls = np.array(normalized_controls)

    if normalized_controls.ndim == 1:
        # Single prediction
        throttle_norm, steering_norm = normalized_controls
        throttle = (throttle_norm + 1) / 2 * (
            THROTTLE_RANGE[1] - THROTTLE_RANGE[0]
        ) + THROTTLE_RANGE[0]
        steering = (steering_norm + 1) / 2 * (
            STEERING_RANGE[1] - STEERING_RANGE[0]
        ) + STEERING_RANGE[0]
        return np.array([throttle, steering])
    else:
        # Batch of predictions
        throttle_norm = normalized_controls[..., 0]
        steering_norm = normalized_controls[..., 1]
        throttle = (throttle_norm + 1) / 2 * (
            THROTTLE_RANGE[1] - THROTTLE_RANGE[0]
        ) + THROTTLE_RANGE[0]
        steering = (steering_norm + 1) / 2 * (
            STEERING_RANGE[1] - STEERING_RANGE[0]
        ) + STEERING_RANGE[0]
        return np.stack([throttle, steering], axis=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class LidarTransformer(nn.Module):
    def __init__(
        self, n_points=7, d_model=128, nhead=8, num_layers=3, dim_feedforward=512
    ):
        super().__init__()

        # Point embedding
        self.point_embedding = nn.Linear(2, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Output heads for steering and throttle
        self.control_head = nn.Sequential(
            nn.Linear(d_model * n_points, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [throttle, steering]
        )

    def forward(self, x):
        # x shape: (batch_size, n_points, 2)

        # Embed each point
        x = self.point_embedding(x)  # (batch_size, n_points, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transform
        x = self.transformer_encoder(x)  # (batch_size, n_points, d_model)

        # Flatten and predict controls
        x = x.reshape(x.size(0), -1)  # (batch_size, n_points * d_model)
        controls = self.control_head(x)  # (batch_size, 2)

        return controls
