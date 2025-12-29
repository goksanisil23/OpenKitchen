import torch
import numpy as np
import json
from train_vae import VAE
from train_rnn import GRUDynamics

# 1. EXPORT VAE
print("Exporting VAE...")
# vae_config = json.load(open("vae_outputs/args.json"))
vae_config = json.load(open("vae_outputs_32/args.json"))
img_size = vae_config["image_size"]
z_dim_vae = vae_config["latent_dim"]
vae_model = VAE(img_size=img_size, z_dim=z_dim_vae, ch=vae_config["base_ch"])
# vae_model.load_state_dict(torch.load("vae_outputs/vae_state.pt", map_location="cpu"))
vae_model.load_state_dict(
    torch.load("vae_outputs_32/vae_state_43.pt", map_location="cpu")
)
vae_model.eval()

# We export the encoder and decoder separately
scripted_encoder = torch.jit.script(vae_model.enc).eval().to("cuda")
scripted_decoder = torch.jit.script(vae_model.dec).eval().to("cuda")

frozen_encoder = torch.jit.freeze(scripted_encoder)
frozen_decoder = torch.jit.freeze(scripted_decoder)

frozen_encoder.save("cpp_models/vae_encoder.pt")
frozen_decoder.save("cpp_models/vae_decoder.pt")
print(f"VAE encoder and decoder saved to cpp_models/")

# 2. EXPORT RNN
print("\nExporting RNN...")
rnn_checkpoint = torch.load("rnn_outputs/gru_dyn_best.pt", map_location="cpu")
rnn_stats = np.load("rnn_outputs/stats.npz")
z_mean = rnn_stats["z_mean"]
z_std = rnn_stats["z_std"]
a_mean = rnn_stats["a_mean"]
a_std = rnn_stats["a_std"]
z_dim_rnn = z_mean.shape[1]
a_dim = a_mean.shape[1]
rnn_args = rnn_checkpoint["args"]
rnn_model = GRUDynamics(
    z_dim=z_dim_rnn,
    a_dim=a_dim,
    hidden=rnn_args["hidden"],
    layers=rnn_args["layers"],
    dropout=rnn_args["dropout"],
)
rnn_model.load_state_dict(rnn_checkpoint["model"])
rnn_model.eval()
rnn_model.gru.flatten_parameters()

scripted_rnn = torch.jit.script(rnn_model).eval().to("cuda")
frozen_rnn = torch.jit.freeze(scripted_rnn)
frozen_rnn.save("cpp_models/rnn_model.pt")
print("RNN model saved to cpp_models/")

# 3. EXPORT STATS
print("\nExporting normalization stats...")
with open("cpp_models/stats.txt", "w") as f:
    f.write(" ".join(map(str, z_mean.flatten())) + "\n")
    f.write(" ".join(map(str, z_std.flatten())) + "\n")
    f.write(" ".join(map(str, a_mean.flatten())) + "\n")
    f.write(" ".join(map(str, a_std.flatten())) + "\n")
print("Stats saved to cpp_models/stats.txt")
