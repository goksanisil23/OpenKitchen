import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf

TRAIN_IMGS_FOLDER = "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/"
TEST_IMGS_FOLDER = "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/"
# MODEL_PATH = "low_res_models/5000.torch"  # Path to trained model
MODEL_PATH = "5000.torch"  # Path to trained model

train_images = glob.glob(TRAIN_IMGS_FOLDER + "/*.png")
test_images = glob.glob(TEST_IMGS_FOLDER + "/*.png")
np.set_printoptions(threshold=np.inf)
height = width = 600

transformImg = tf.Compose(
    [
        tf.ToPILImage(),
        tf.Resize((height, width)),
        tf.ToTensor(),
        # tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Check if there is GPU if not set trainning to CPU (very slow)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
# Change final layer to 3 classes
Net.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(MODEL_PATH))  # Load trained model
Net.eval()  # Set to evaluation mode

for img_path in test_images:
    # for img_path in train_images:
    Img = cv2.imread(img_path)  # load test image
    height_orgin, widh_orgin, d = Img.shape  # Get image original size
    # Remove the annotated layer from the training image
    Img[(Img[:, :, 0] == 0) & (Img[:, :, 1] == 255) & (Img[:, :, 2] == 0)] = [0, 0, 0]
    cv2.imshow("img", Img)
    cv2.waitKey(10)
    Img = transformImg(Img)  # Transform to pytorch
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
        Prd = Net(Img)["out"]  # Run net
    Prd = tf.Resize((height_orgin, widh_orgin))(Prd[0])  # Resize to origninal size
    # Prd = (3,600,600), so each pixel gets 3 probability score (1 for each available class), we choose the max of them
    seg = torch.argmax(Prd, 0).cpu().detach().numpy().astype(np.uint8)
    # Convert to a more distinct grayscale image based on pixel-wise classes
    seg[seg == 1] = 250
    seg[seg == 2] = 250
    seg[seg == 3] = 120
    cv2.imshow("seg", seg)
    cv2.waitKey()
