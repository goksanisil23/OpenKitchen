import os

import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions"

filenames = [f[:-4] for f in os.listdir(DIRECTORY) if f.endswith(".png")]

angles = []

for filename in filenames:
    action_path = os.path.join(DIRECTORY, filename + ".txt")
    with open(action_path, "r") as f:
        action = list(map(float, f.read().strip().split()))
        angles.append(action[1])
        if abs(action[1]) > 180.0:
            print(filename)
        if action[0] > 100.0:
            print(filename)

plt.stem(angles)
plt.show()
