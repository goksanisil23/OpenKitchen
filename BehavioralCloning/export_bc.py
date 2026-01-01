# export_torchscript.py
import torch
import torch.nn as nn

from env_train_bc import PolicyCNN


def main():
    ckpt = "sao_paulo_policy.pt"
    out = "sao_paulo_policy_ts.pt"

    device = torch.device("cpu")

    model = PolicyCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(out)
    print(f"Saved scripted model to: {out}")

    # Equivalence check
    torch.manual_seed(0)
    max_abs_err = 0.0
    for i in range(10):
        x = torch.rand(1, 3, 96, 96)
        with torch.no_grad():
            y_ref = model(x)
            y_ts = scripted_model(x)
        abs_err = (y_ref - y_ts).abs().max().item()
        max_abs_err = max(max_abs_err, abs_err)

    print(f"Max absolute error over 10 random inputs: {max_abs_err:.6e}")
    assert max_abs_err < 1e-6, "Scripted model outputs differ from original model!"


if __name__ == "__main__":
    main()
