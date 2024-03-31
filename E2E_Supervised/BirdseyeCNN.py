import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdseyeCNN(nn.Module):
    def __init__(self, grayscale=False, steering_only=False, downscale_factor=1):
        super(BirdseyeCNN, self).__init__()

        input_img_dim = 1 if grayscale else 3
        output_action_dim = 1 if steering_only else 2

        # Output size = ((input_size - kernel_size+2*padding)/stride)+1
        self.conv1 = nn.Conv2d(input_img_dim, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        if downscale_factor == 1:
            self.fc1 = nn.Linear(64 * 193 * 168, 100)  # when 1600, 1400
        elif downscale_factor == 4:
            self.fc1 = nn.Linear(64 * 37 * 43, 100)  # when 1600//4, 1400/4
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_action_dim)  # Output: steering and acceleration

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)  # Flatten the tensors for the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Using tanh to get outputs in the range [-1, 1]

        return x
