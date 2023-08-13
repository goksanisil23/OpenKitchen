import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf

TEST_IMGS_FOLDER = (
    "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/test/"
)
INPUT_MODEL_PATH = "5000.torch"  # Path to trained model
OUTPUT_ONNX_MODEL = "semseg.onnx"

# test_images = glob.glob(TEST_IMGS_FOLDER + "/*.png")
test_images = [
    "/home/s0001734/Downloads/OpenKitchen/SemSegRacer/raylib_images/00004.png"
]
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
Net.load_state_dict(torch.load(INPUT_MODEL_PATH))  # Load trained model
Net.eval()  # Set to evaluation mode

# Create dummy input and export to ONNX format
dummy_input = torch.randn(1, 3, height, width).to(device)
torch.onnx.export(Net, dummy_input, OUTPUT_ONNX_MODEL)

# Load ONNX Model
ort_session = ort.InferenceSession(OUTPUT_ONNX_MODEL)


for img_path in test_images:
    Img = cv2.imread(img_path)  # load test image
    height_orgin, widh_orgin, d = Img.shape  # Get image original size
    # Remove the annotated layer from the training image
    Img[(Img[:, :, 0] == 0) & (Img[:, :, 1] == 255) & (Img[:, :, 2] == 0)] = [0, 0, 0]
    Img = transformImg(Img)  # Transform to pytorch

    Img = Img.numpy().astype(np.float32)  # convert to numpy array
    Img = np.expand_dims(Img, axis=0)  # Add batch dimension
    # print(Img.ravel()[500:600])

    # Inference
    ort_inputs = {ort_session.get_inputs()[0].name: Img}
    # Run model, [0] to get the output tensor
    Prd = ort_session.run(None, ort_inputs)[0]

    # Resize to original size
    Prd = tf.Resize((height_orgin, widh_orgin))(torch.from_numpy(Prd[0]))

    # Get the predicted class for each pixel
    seg = torch.argmax(Prd, 0).numpy().astype(np.uint8)
    seg[seg == 1] = 250
    seg[seg == 2] = 250
    seg[seg == 3] = 120
    cv2.imshow("seg", seg)
    cv2.waitKey()
