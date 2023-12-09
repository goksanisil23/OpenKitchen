import matplotlib.pyplot as plt
import torch
from laser_2d_autoencoder import Autoencoder, ScanDataset
from torch.utils.data import DataLoader

folder_path = "collect_data_racetrack/build/point_clouds/"
dataset = ScanDataset(folder_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(torch.load("laser_2d_autoencoder.pth"))
model.eval()

############## TESTING ##################
while True:
    test_indices = torch.randint(0, len(dataset), (5,))

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Original vs Reconstructed Points")
    for i, idx in enumerate(test_indices):
        original = dataset[idx].to(device).unsqueeze(0)  # Add batch dimension

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            reconstructed = model(original).cpu()

        # Reshape data for plotting (5 2D points)
        original = original.view(-1, 2).cpu().numpy()
        reconstructed = reconstructed.view(-1, 2).cpu().numpy()

        # Plot original points
        axes[0, i].scatter(original[:, 0], original[:, 1], color="blue")
        axes[0, i].set_title(f"Original {idx}")
        axes[0, i].axis("equal")

        # Plot reconstructed points
        axes[1, i].scatter(reconstructed[:, 0], reconstructed[:, 1], color="red")
        axes[1, i].set_title(f"Reconstructed {idx}")
        axes[1, i].axis("equal")

    plt.show()
