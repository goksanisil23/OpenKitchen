import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 6),  # compress to 3 features
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # output a value between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Load the full model
model = torch.load("full_autoencoder_model.pth")
model.eval()  # Set the model to evaluation mode

# Visualize the weights of the first layer of the encoder
with torch.no_grad():
    # Assuming the first layer in your encoder is a nn.Linear layer
    weights = (
        model.encoder[0].weight.data.view(128, 28, 28).cpu()
    )  # Reshape weights to 28x28

plt.figure(figsize=(15, 15))
for i in range(128):  # Assuming there are 128 neurons in the first layer
    plt.subplot(12, 12, i + 1)  # Adjust the subplot grid if needed
    plt.imshow(weights[i], cmap="gray")
    plt.axis("off")
plt.show()
