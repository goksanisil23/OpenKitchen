import torch
from BirdseyeCNN import BirdseyeCNN
from data_loaders.birdseye_data_loader import SensorMeasurementsDataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Instantiate the dataset
data_directory_path = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions"
img_transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda img: transforms.functional.resize(
                img, (img.height // 4, img.width // 4)
            )
        ),
        transforms.ToTensor(),
    ]
)
print("Loading dataset...")
dataset = SensorMeasurementsDataset(data_directory_path, img_transform=img_transform)

# Splitting the dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)
print(f"train dataset size: {train_size}")
print(f"validation dataset size: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model, Loss Function, and Optimizer
model = BirdseyeCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
if torch.cuda.is_available():
    model.cuda()

print("Starting training")
# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (measurement, action) in enumerate(train_loader):
        if torch.cuda.is_available():
            measurement, action = measurement.cuda(), action.cuda()  # Move data to GPU
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = model(measurement)
        loss = loss_fn(outputs, action)
        loss.backward()
        optimizer.step()
        # print(f"batch {batch_idx} done")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (measurement, action) in enumerate(val_loader):
            if torch.cuda.is_available():
                measurement, action = measurement.cuda(), action.cuda()  # Move data
            action_prediction = model(measurement)
            val_loss += loss_fn(action_prediction, action).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Loss: {loss.item():.5f}, Val Loss: {val_loss:.5f}")
    torch.save(model.state_dict(), "birdseye_cnn.pth")


# Save the model
torch.save(model.state_dict(), "birdseye_cnn.pth")
