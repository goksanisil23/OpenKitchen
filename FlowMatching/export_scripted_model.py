import argparse
from pathlib import Path

import torch

from flow_matching_model import ConditionalFlowMatchingPolicy


@torch.no_grad()
def verify_equivalence(
    eager_model: torch.nn.Module,
    scripted_model: torch.jit.ScriptModule,
    image_size: int,
    device: torch.device,
    num_tests: int,
    batch_size: int,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> float:
    max_abs_diff = 0.0

    eager_model.eval()
    scripted_model.eval()

    for _ in range(num_tests):
        x_t = torch.randn(batch_size, 2, device=device)
        t = torch.rand(batch_size, 1, device=device)
        image = torch.rand(batch_size, 3, image_size, image_size, device=device)

        eager_out = eager_model(x_t, t, image)
        scripted_out = scripted_model(x_t, t, image)

        current_max_abs_diff = (eager_out - scripted_out).abs().max().item()
        max_abs_diff = max(max_abs_diff, current_max_abs_diff)

        if not torch.allclose(eager_out, scripted_out, atol=atol, rtol=rtol):
            raise RuntimeError(
                "Scripted model output does not match eager model output "
                f"(max_abs_diff={current_max_abs_diff:.8f})."
            )

    return max_abs_diff


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Export trained conditional flow matching model to TorchScript."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to training checkpoint (.pt) containing model_state_dict and config.",
    )
    parser.add_argument(
        "--output-scripted",
        type=str,
        default="checkpoints/policy_scripted.pt",
        help="Path where scripted model (.pt) will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for export/verification.",
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=5,
        help="Number of random batches used for output equivalence checks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during output equivalence checks.",
    )
    return parser


def main():
    args = build_argparser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    image_size = int(config["image_size"])

    model = ConditionalFlowMatchingPolicy(
        bev_dim=int(config["bev_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scripted_model = torch.jit.script(model)

    max_abs_diff = verify_equivalence(
        eager_model=model,
        scripted_model=scripted_model,
        image_size=image_size,
        device=device,
        num_tests=args.num_tests,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output_scripted)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_model.save(str(output_path))

    print(f"Saved scripted model to: {output_path}")
    print(
        "Verification passed: scripted model matches eager model "
        f"(max_abs_diff={max_abs_diff:.8f})."
    )


if __name__ == "__main__":
    main()
