import numpy as np
import torch
from PIL import Image
import open_kitchen_pybind as ok

import env_train_bc as train

SIZE = (96, 96)
TRACK = "/home/s0001734/Downloads/racetrack-database/tracks/SaoPaulo.csv"
# TRACK = "/home/s0001734/Downloads/racetrack-database/tracks/Sakhir.csv"
# MODEL = "sao_paulo_policy.pt"
MODEL = "sao_paulo_policy_ts.pt"


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    # Minimal input formatting (no model duplication): HWC uint8 -> NCHW float32 in [0,1]
    x = obs.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (3,H,W)
    return torch.from_numpy(x).unsqueeze(0).to(device)  # (1,3,H,W)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = train.PolicyCNN().to(device)
    # net.load_state_dict(torch.load(MODEL, map_location=device))
    net = torch.jit.load(MODEL, map_location=device)
    net.eval()

    env = ok.Environment(
        TRACK,
        draw_rays=False,
        hidden_window=False,
    )

    with torch.inference_mode():
        while True:
            env.step()
            img = np.frombuffer(bytes(env.get_render_target()), dtype=np.uint8)
            info = env.get_render_target_info()
            img = np.ascontiguousarray(
                np.flip(img.reshape(info.height, info.width, info.channels), axis=0)
            )
            img = np.asarray(
                Image.fromarray(img).resize(SIZE, resample=Image.Resampling.BICUBIC),
                dtype=np.uint8,
            )
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]

            input_tensor = obs_to_tensor(img, device)
            action = net(input_tensor).squeeze(0).cpu().numpy()
            steer_raw = float(action) * 10.0  # scale steering
            env.set_action(100.0, steer_raw)

            # debug_img = (input_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
            #     np.uint8
            # )
            # Image.fromarray(debug_img).save("input_tensor_preview.png")


if __name__ == "__main__":
    main()
