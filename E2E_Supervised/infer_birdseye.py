import struct
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np
import posix_ipc
import torch
import torch.nn as nn
import torch.nn.functional as F
from BirdseyeCNN import BirdseyeCNN
from PIL import Image
from torchvision import transforms

GRAYSCALE = True
STEER_ONLY = True
SHOW_IMG = False
IMG_WIDTH = 1600
IMG_HEIGTH = 1400
IMG_SIZE = IMG_WIDTH * IMG_HEIGTH * 4  # RGBA
DOWNSCALE_FACTOR = 4
ANGLE_DENORMALIZATION = 1.63364
MODEL_PATH = "birdseye_cnn.pth"

shm_meas = shared_memory.SharedMemory(name="shm_measurements")
shm_actions = shared_memory.SharedMemory(name="shm_actions")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BirdseyeCNN(
    grayscale=GRAYSCALE, steering_only=STEER_ONLY, downscale_factor=DOWNSCALE_FACTOR
)
model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


img_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (IMG_HEIGTH // DOWNSCALE_FACTOR, IMG_WIDTH // DOWNSCALE_FACTOR)
        ),
        transforms.ToTensor(),
    ]
)

with torch.no_grad():
    while True:  # Continuous loop
        # Wait for C++ to signal that data is ready
        sem1.acquire()

        # Read data from shared memory
        data = shm_meas.buf[:IMG_SIZE]  # Adjust size based on your data structure
        # only take RGB of RGBA
        image_array_from_shm = np.frombuffer(data, dtype=np.uint8).reshape(
            IMG_HEIGTH, IMG_WIDTH, 4
        )[:, :, :3]
        if SHOW_IMG:
            Image.fromarray(image_array_from_shm).convert("L").show()
        if GRAYSCALE:
            image_array_from_shm = np.array(
                Image.fromarray(image_array_from_shm).convert("L")
            )

        # Create tensor from the received data
        image_tensor = img_transform(image_array_from_shm).unsqueeze(0).to(device)

        # # Infer the model
        action = model(image_tensor).cpu()

        # # Send the response, after denormalization
        if STEER_ONLY:
            response = (100.0, action.numpy().flatten()[0] * ANGLE_DENORMALIZATION)
        else:
            response = tuple(action.numpy().flatten())
            response[0] *= 100.0
            response[1] *= ANGLE_DENORMALIZATION
        print(f"denorm. action: {response}")
        shm_actions.buf[0:8] = struct.pack("ff", *response)

        # # Signal C++ that response is ready
        sem2.release()


example_input = torch.rand((1, 7, 2))  # Shape: (batch_size, sequence_length, input_dim)
