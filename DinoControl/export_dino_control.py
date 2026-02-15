import torch
import torch.nn as nn
import timm

from train_dino_control import MODEL_NAME, PolicyHead


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.embed_dim = self.backbone.num_features
        self.head = PolicyHead(self.embed_dim)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def main():
    WEIGHTS_PATH = "agent_model_epoch_50.pth"
    EXPORT_PATH = "agent_model_scripted.pt"

    device = torch.device("cpu")

    print("Initializing model...")
    # Initialize the model structure with scriptable=True
    model = Agent().to(device)

    print(f"Loading weights from {WEIGHTS_PATH}...")
    # Load the weights you trained
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Script the model
    print("Scripting model (torch.jit.script)...")
    # No dummy input needed for scripting; it compiles the code directly
    scripted_module = torch.jit.script(model)

    # 4. Save
    scripted_module.save(EXPORT_PATH)
    print(f"Success! Model exported to {EXPORT_PATH}")

    # 5. Verify Output
    print("Verifying...")
    dummy_input = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        orig_out = model(dummy_input)
        script_out = scripted_module(dummy_input)

        if torch.allclose(orig_out, script_out, atol=1e-5):
            print("Verification PASSED: Outputs match.")
        else:
            print("Verification FAILED: Outputs differ.")


if __name__ == "__main__":
    main()
