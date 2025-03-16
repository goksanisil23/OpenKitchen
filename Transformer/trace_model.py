import torch
from laser_transformer import LidarTransformer

# Load and prepare the original model
model = LidarTransformer(n_points=7)
checkpoint = torch.load("best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Create a dummy input (normalized laser points)
dummy_input = torch.randn(1, 7, 2)

# Get output from the original model
orig_output = model(dummy_input)

# Trace the model using TorchScript
traced_model = torch.jit.trace(model, dummy_input,check_trace=False)
traced_output = traced_model(dummy_input)

print(f"Original output: {orig_output}")
print(f"Traced output: {traced_output}")

# Compare outputs (using a tolerance to account for minor differences)
if torch.allclose(orig_output, traced_output, atol=1e-6):
    print("Outputs match!")
    traced_model.save("best_model_scripted.pt")
else:
    print("Outputs differ!")
