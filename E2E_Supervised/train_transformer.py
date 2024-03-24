import torch
from data_loaders.laser_data_loader import SensorMeasurementsDataset
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from Transformer import TransformerModel

# Instantiate the dataset
data_directory_path = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/pointcloud_and_actions"
dataset = SensorMeasurementsDataset(data_directory_path)
# Splitting the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, Loss Function, and Optimizer
model = TransformerModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_idx, (measurement, action) in enumerate(train_loader):

        # Forward pass
        action_prediction = model(measurement)
        loss = loss_fn(action_prediction, action)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (measurement, action) in enumerate(val_loader):
            action_prediction = model(measurement)
            val_loss += loss_fn(action_prediction, action).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "simple_transformer.pth")
