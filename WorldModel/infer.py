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
from train_rnn import GRUDynamics
from collections import deque


IMG_WIDTH = 1600
IMG_HEIGTH = 1400
IMG_SIZE = IMG_WIDTH * IMG_HEIGTH * 4  # RGBA


STRIDE = 10  # Important! If we every frame without sufficient displacement, RNN fails since the training data is mainly 100 acceleration


def to_uint8(t):
    t = t.squeeze(0).clamp(0, 1).detach().cpu().numpy()  # CxHxW
    t = np.transpose(t, (1, 2, 0))  # HxWxC, RGB
    t = (t * 255.0).round().astype(np.uint8)
    return t


def z_norm(z):  # z: (1,1,dz)
    return (z - z_mean_b) / z_std_b


def a_norm(a):  # a: (1,1,da)
    return (a - a_mean_b) / a_std_b


def z_denorm(z):  # z: (1,1,dz)
    return z * z_std_b + z_mean_b


#  ----- shard mem setup -----
shm_meas = shared_memory.SharedMemory(name="shm_measurements")
shm_actions = shared_memory.SharedMemory(name="shm_actions")
sem1 = posix_ipc.Semaphore("/sem1", posix_ipc.O_CREAT, initial_value=0)
sem2 = posix_ipc.Semaphore("/sem2", posix_ipc.O_CREAT, initial_value=0)

# ----- VAE setup -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model_config = json.load(open("vae_outputs/args.json"))
vae_state_dict = torch.load("vae_outputs/vae_state.pt", map_location="cpu")
vae_model = VAE(
    img_size=vae_model_config["image_size"],
    z_dim=vae_model_config["latent_dim"],
    ch=vae_model_config["base_ch"],
).to(device)
vae_model.load_state_dict(vae_state_dict)
vae_model.eval()

img_size = getattr(
    getattr(vae_model, "dec", vae_model), "img_size", vae_model_config["image_size"]
)
trans_img = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1], CxHxW
    ]
)

# ----- RNN setup -----
rnn_checkpoint = torch.load("dyn_outputs/gru_dyn_best.pt", map_location="cpu")
rnn_stats = np.load("dyn_outputs/stats.npz")
z_mean = torch.from_numpy(rnn_stats["z_mean"]).float().to(device)  # (1, dz)
z_std = torch.from_numpy(rnn_stats["z_std"]).float().to(device)
a_mean = torch.from_numpy(rnn_stats["a_mean"]).float().to(device)  # (1, da)
a_std = torch.from_numpy(rnn_stats["a_std"]).float().to(device)

z_dim = z_mean.shape[1]
a_dim = a_mean.shape[1]

rnn_args = rnn_checkpoint["args"]
rnn_model = GRUDynamics(
    z_dim=z_dim,
    a_dim=a_dim,
    hidden=rnn_args["hidden"],
    layers=rnn_args["layers"],
    dropout=rnn_args["dropout"],
).to(device)
rnn_model.load_state_dict(rnn_checkpoint["model"])
rnn_model.eval()

# pre-broadcast stats to (1,1,dim) for (B=1,T=1,dim) ops
z_mean_b = z_mean.unsqueeze(1)  # (1,1,dz)
z_std_b = z_std.unsqueeze(1)
a_mean_b = a_mean.unsqueeze(1)  # (1,1,da)
a_std_b = a_std.unsqueeze(1)


# keep recurrent state across steps
h = None

vae_window = "VAE Reconstructions"
rnn_window = "RNN pred"
cv2.namedWindow(vae_window, cv2.WINDOW_NORMAL)
cv2.namedWindow(rnn_window, cv2.WINDOW_NORMAL)

seq_len = 1
x_buf = deque(maxlen=seq_len)

acc = 100
step = 0
with torch.no_grad():
    while True:  # Continuous loop

        # Wait for simulator to signal that data is ready
        sem1.acquire()

        data = shm_meas.buf[:IMG_SIZE]
        # only take RGB of RGBA
        image_array_from_shm = np.frombuffer(data, dtype=np.uint8).reshape(
            IMG_HEIGTH, IMG_WIDTH, 4
        )[:, :, :3]

        # Transform the image for VAE network
        vae_input = trans_img(image_array_from_shm).unsqueeze(0).to(device)  # 1xCxHxW
        recon_t, mu_t, logvar_t = vae_model(vae_input)
        z_t = mu_t.view(1, 1, -1)
        vae_rec_t = to_uint8(recon_t)

        cv2.imshow(vae_window, vae_rec_t)
        # cv2.imwrite(f"vae_recon_{step:04d}.png", vae_rec_t)

        response = (acc, 0.0)

        if step % STRIDE == 0:
            a_t = torch.tensor(response, dtype=torch.float32, device=device).view(
                1, 1, -1
            )
            # GRU single-step: (z_t, a_t) -> z_{t+1}
            x_t = torch.cat([z_norm(z_t), a_norm(a_t)], dim=-1)  # (1,1,dz+da)
            x_buf.append(x_t)

            if len(x_buf) == seq_len:
                x_seq = torch.cat(list(x_buf), dim=1)
                z_next_pred_normalized, _ = rnn_model(x_seq)

                z_next = z_denorm(z_next_pred_normalized).view(1, -1)

                # Decode the predicted z_{t+1}
                pred = vae_model.decode(z_next)
                pred_img = to_uint8(pred)

                cv2.imshow(rnn_window, pred_img)
                # cv2.imwrite(f"rnn_pred_{step:04d}.png", pred_img)
            cv2.waitKey(1)

        # -------- Actions --------
        print(f"denorm. action: {response}")
        shm_actions.buf[0:8] = struct.pack("ff", *response)

        # # Signal C++ that response is ready
        sem2.release()

        step += 1


cv2.destroyAllWindows()
