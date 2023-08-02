"""
This script is for training a semantic segmentation model based on race track images rendered via Raylib

input a set of race track images, each paired with a corresponding segmentation mask. 
The model is trained to classify each pixel in the image according to the different 
classes represented in the masks (e.g., road, grass, sand, etc.).

Usage:
    python3 train_segmentation.py --images path/to/images --masks path/to/masks

Inputs:
    --images: A directory containing the input images. The images should be in JPEG or PNG format.
    --masks: A directory containing the corresponding segmentation masks. Each mask should have the same name as its corresponding image.

Outputs:
    A trained model file, as well as plots of the model's performance over time.
"""

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf

torch.cuda.empty_cache()

Learning_Rate = 1e-5
WIDTH = HEIGHT = 600  # desired image width and height for training
BATCH_SIZE = 3
EPOCH_LIMIT = 5000

train_dataset_folder = "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images"
train_images = glob.glob(train_dataset_folder + "/*.png")

# ----------------------------------------------Transform image-------------------------------------------------------------------
transformImg = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((HEIGHT, WIDTH)),
        tf.ToTensor(),
        # tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
transformAnn = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((HEIGHT, WIDTH), tf.InterpolationMode.NEAREST),
        tf.ToTensor(),
    ]
)


def generateAnnotationForRaylibImg(rgb_img: np.ndarray):
    # Set all annotation pixels to 0 by default
    anno_mask = np.zeros(rgb_img.shape[0:2], np.float32)
    # Pure red pixels are right(anno = 1), pure blue pixels are left (anno = 2) lane
    anno_mask[
        np.where(
            (rgb_img[:, :, 0] == 0)
            & (rgb_img[:, :, 1] == 0)
            & (rgb_img[:, :, 2] == 255)
        )
    ] = 1

    anno_mask[
        np.where(
            (rgb_img[:, :, 0] == 255)
            & (rgb_img[:, :, 1] == 0)
            & (rgb_img[:, :, 2] == 0)
        )
    ] = 2

    return anno_mask


# ---------------------Read image ---------------------------------------------------------
def ReadRandomImage():  # First lets load random image and  the corresponding annotation
    idx = np.random.randint(0, len(train_images))  # Select random image
    rgb_img = cv2.imread(train_images[idx])[:, :, 0:3]

    anno_mask = generateAnnotationForRaylibImg(rgb_img)

    rgb_tensor = transformImg(rgb_img)
    anno_tensor = transformAnn(anno_mask)
    return rgb_tensor, anno_tensor


# --------------Load batch of images-----------------------------------------------------
def LoadBatch():  # Load batch of images
    images = torch.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH])
    ann = torch.zeros([BATCH_SIZE, HEIGHT, WIDTH])
    for i in range(BATCH_SIZE):
        images[i], ann[i] = ReadRandomImage()
    return images, ann


# --------------Load and set net and optimizer-------------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Load neuralnet
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# Change final layer to 3 classes
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
Net = Net.to(device)
# Create adam optimizer
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)


# ----------------Train--------------------------------------------------------------------------
for itr in range(EPOCH_LIMIT + 1):  # Training loop
    images, ann = LoadBatch()  # Load taining batch
    images = torch.autograd.Variable(images, requires_grad=False).to(
        device
    )  # Load image batch
    ann = torch.autograd.Variable(ann, requires_grad=False).to(
        device
    )  # Load annotation batch
    Pred = Net(images)["out"]  # make prediction
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss()  # Set loss function
    Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
    Loss.backward()  # Backpropogate loss
    optimizer.step()  # Apply gradient descent change to weight
    seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get  prediction classes
    print(f"iter: {itr} Loss: {Loss.data.cpu().numpy()}")
    if itr % 1000 == 0:  # Save model weight once every 1000 epochs
        print("Saving Model" + str(itr) + ".torch")
        torch.save(Net.state_dict(), str(itr) + ".torch")
