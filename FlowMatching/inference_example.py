import argparse
from pathlib import Path

import torch

from flow_matching_model import ActionNormalizer, ConditionalFlowMatchingPolicy, preprocess_bev_image


@torch.no_grad()
def sample_action(model, image_tensor, num_steps, device):
    """Integrates dx/dt = v_theta(x, t, cond) from random noise to final action."""
    model.eval()

    x = torch.randn(1, 2, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t_val = i / num_steps
        t = torch.full((1, 1), t_val, device=device)
        v = model(x_t=x, t=t, image=image_tensor)
        x = x + dt * v

    return x


def build_argparser():
    parser = argparse.ArgumentParser(description="Run inference with trained conditional flow matching policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to a BEV image (.png)")
    parser.add_argument("--num-steps", type=int, default=32, help="Euler integration steps for ODE sampling")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    normalizer_cfg = ckpt.get("normalizer", {})
    normalizer = ActionNormalizer(
        throttle_min=normalizer_cfg.get("throttle_min", 0.0),
        throttle_max=normalizer_cfg.get("throttle_max", 100.0),
        steering_min=normalizer_cfg.get("steering_min", -10.0),
        steering_max=normalizer_cfg.get("steering_max", 10.0),
    )

    model = ConditionalFlowMatchingPolicy(
        bev_dim=config.get("bev_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    image_tensor = preprocess_bev_image(Path(args.image), image_size=config.get("image_size", 128)).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    action_norm = sample_action(model, image_tensor=image_tensor, num_steps=args.num_steps, device=device)
    action_norm = action_norm.squeeze(0).cpu().clamp(-1.0, 1.0)

    throttle_norm = float(action_norm[0])
    steering_norm = float(action_norm[1])
    throttle, steering = normalizer.denormalize(throttle_norm, steering_norm)

    print("Predicted normalized action:")
    print(f"  throttle_norm: {throttle_norm:.4f}")
    print(f"  steering_norm: {steering_norm:.4f}")

    print("Predicted physical action:")
    print(f"  throttle: {throttle:.2f} (expected range 0..100)")
    print(f"  steering: {steering:.2f} deg (expected range -10..10)")


if __name__ == "__main__":
    main()
