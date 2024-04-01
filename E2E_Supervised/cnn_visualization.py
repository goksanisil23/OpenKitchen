import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from BirdseyeCNN import BirdseyeCNN
from PIL import Image
from torchvision import transforms
from torchviz import make_dot

GRAYSCALE = True
STEER_ONLY = True
SHOW_IMG = False
IMG_WIDTH = 1600
IMG_HEIGTH = 1400
IMG_SIZE = IMG_WIDTH * IMG_HEIGTH * 4  # RGBA
DOWNSCALE_FACTOR = 4
MODEL_PATH = "birdseye_cnn.pth"


# Function to get the feature maps
def get_feature_maps(model, input_image, layer_num):

    x = F.relu(model.conv1(input_image))
    if layer_num == 1:
        return x

    x = F.relu(model.conv2(x))
    if layer_num == 2:
        return x

    x = F.relu(model.conv3(x))
    if layer_num == 3:
        return x

    x = F.relu(model.conv4(x))
    if layer_num == 4:
        return x

    x = F.relu(model.conv5(x))
    if layer_num == 5:
        return x


# Visualization function for the feature maps
def visualize_feature_maps(feature_maps):
    feature_maps = feature_maps.squeeze(0)  # remove the batch dim
    num_feature_maps = feature_maps.shape[0]

    # Calculate the number of rows and columns to display the feature maps
    num_cols = np.ceil(np.sqrt(num_feature_maps)).astype(int)
    num_rows = np.ceil(num_feature_maps / num_cols).astype(int)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            ax.imshow(feature_maps[i].detach().cpu().numpy(), cmap="gray")
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()


model = BirdseyeCNN(
    grayscale=GRAYSCALE, steering_only=STEER_ONLY, downscale_factor=DOWNSCALE_FACTOR
)

image = Image.open(
    "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions_IMS/birdseye_IMS_208_0.png"
)
np_image = np.array(image)

img_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(
            (IMG_HEIGTH // DOWNSCALE_FACTOR, IMG_WIDTH // DOWNSCALE_FACTOR)
        ),
        transforms.ToTensor(),
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_array = np.array(Image.fromarray(np_image).convert("L"))

image_tensor = img_transform(image_array).unsqueeze(0).to(device)

model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

feature_maps = get_feature_maps(model, image_tensor, layer_num=4)
visualize_feature_maps(feature_maps)
