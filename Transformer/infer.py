import struct
from multiprocessing import shared_memory

import numpy as np
import posix_ipc
import torch
import torch.nn as nn
from laser_transformer import (
    LidarTransformer,
    denormalize_controls,
    normalize_controls,
    normalize_laser_point,
)

# 7 laser points, each with 2 coordinates (x, y), 4 bytes each
MEAUSUREMENTS_SIZE = 7 * 2 * 4

shm_meas = shared_memory.SharedMemory(name="shm_measurements")
shm_actions = shared_memory.SharedMemory(name="shm_actions")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = LidarTransformer(n_points=7)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

np.set_printoptions(suppress=True, precision=4)

# Get prediction
with torch.no_grad():
    while True:
        # Wait for C++ to signal that data is ready
        sem1.acquire()

        # Read data from shared memory
        data = shm_meas.buf[:MEAUSUREMENTS_SIZE]
        # unpack data into a tensor
        laser_points = np.array(struct.unpack("14f", data)).reshape(7, 2)
        # Normalize
        laser_points = np.array(
            [normalize_laser_point(point) for point in laser_points]
        )
        laser_tensor = torch.FloatTensor(laser_points).unsqueeze(0).to(device)

        predictions = model(laser_tensor)
        unnorm_pred = denormalize_controls(predictions.cpu().numpy())
        print(f"control prediction {unnorm_pred}")

        response = (unnorm_pred[0][0], unnorm_pred[0][1])
        shm_actions.buf[0:8] = struct.pack("ff", *response)

        # # Signal C++ that response is ready
        sem2.release()

        # print(f"Predicted controls: throttle={throttle:.3f}, steering={steering:.3f}")
