#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


### HYPERPARAMETERS ###
N_EPOCH = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

images, labels = next(iter(train_loader))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model, loss function, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the Autoencoder
epochs = N_EPOCH
for epoch in range(epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        # Move data to GPU
        print(img.size())
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model, "full_autoencoder_model.pth")

################## TESTING #############################
# Load the test dataset
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# Get a batch of test images
test_images, _ = next(iter(test_loader))
test_images = test_images.view(test_images.size(0), -1)
test_images = test_images.to(device)

# Pass the test images through the autoencoder
with torch.no_grad():  # We don't need to track gradients here
    reconstructed = model(test_images).cpu()

# Convert the images back to 28x28 format
test_images = test_images.view(test_images.size(0), 1, 28, 28)
reconstructed = reconstructed.view(reconstructed.size(0), 1, 28, 28)

# Plot the original and reconstructed images
plt.figure(figsize=(20, 4))
for i in range(10):
    # Display original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(test_images[i].cpu().numpy().squeeze(), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].numpy().squeeze(), cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
