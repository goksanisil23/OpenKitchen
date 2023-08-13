import os

import cv2

# Define paths
source_directory = "raylib_images"
target_directory = "raw_images"

# Create target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Loop through all the files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".png"):
        filepath = os.path.join(source_directory, filename)

        # Read the image using OpenCV
        img = cv2.imread(filepath)

        # Replace pure green pixels with black
        mask = (img[:, :, 0] == 0) & (img[:, :, 1] == 255) & (img[:, :, 2] == 0)
        img[mask] = [0, 0, 0]

        # Save the modified image to the target directory
        output_filepath = os.path.join(target_directory, filename)
        cv2.imwrite(output_filepath, img)

print("Processing complete!")
