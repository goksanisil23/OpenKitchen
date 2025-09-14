import struct
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import numpy as np
import posix_ipc
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
from train_vae import VAE

SHOW_IMG = False
IMG_WIDTH = 1600
IMG_HEIGTH = 1400
IMG_SIZE = IMG_WIDTH * IMG_HEIGTH * 4  # RGBA
DOWNSCALE_FACTOR = 4
ANGLE_DENORMALIZATION = 1.63364
MODEL_PATH = "birdseye_cnn.pth"


def to_uint8(t):
    t = t.squeeze(0).clamp(0, 1).detach().cpu().numpy()  # CxHxW
    t = np.transpose(t, (1, 2, 0))  # HxWxC, RGB
    t = (t * 255.0).round().astype(np.uint8)
    return t


shm_meas = shared_memory.SharedMemory(name="shm_measurements")
shm_actions = shared_memory.SharedMemory(name="shm_actions")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = json.load(open("vae_outputs/args.json"))
state_dict = torch.load("vae_outputs/vae_state.pt", map_location="cpu")
model = VAE(
    img_size=model_config["image_size"],
    z_dim=model_config["latent_dim"],
    ch=model_config["base_ch"],
).to(device)
model.load_state_dict(state_dict)
model.eval()

img_size = getattr(getattr(model, "dec", model), "img_size", model_config["image_size"])
xform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1], CxHxW
    ]
)

window_name = "VAE Reconstructions"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL for resizable

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
            # Image.fromarray(image_array_from_shm).convert("L").show()
            opencv_img = image_array_from_shm
            cv2.imshow("frame", opencv_img)
            cv2.waitKey(0)

        # Transform the image
        x = xform(image_array_from_shm).unsqueeze(0).to(device)  # 1xCxHxW
        recon, _, _ = model(x)  # 1xCxHxW
        rec = to_uint8(recon)

        cv2.imshow(window_name, rec)
        cv2.waitKey(0)

        response = (10.0, 0.0)
        print(f"denorm. action: {response}")
        shm_actions.buf[0:8] = struct.pack("ff", *response)

        # # Signal C++ that response is ready
        sem2.release()


cv2.destroyAllWindows()
