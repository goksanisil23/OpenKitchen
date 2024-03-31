import os

import matplotlib.pyplot as plt
import numpy as np

DIRECTORY = "/home/s0001734/Downloads/OpenKitchen/FieldNavigators/collect_data/build/measurements_and_actions_IMS"

filenames = [f[:-4] for f in os.listdir(DIRECTORY) if f.endswith(".png")]

angles = []
accs = []

for filename in filenames:
    action_path = os.path.join(DIRECTORY, filename + ".txt")
    with open(action_path, "r") as f:
        action = list(map(float, f.read().strip().split()))
        accs.append(action[0])
        angles.append(action[1])
        if abs(action[1]) > 180.0:
            print(filename)
        if action[0] > 100.0:
            print(filename)

print("Mean:", np.mean(angles))
print("Variance:", np.var(angles))

# plt.stem(angles)
plt.hist(angles, bins=np.arange(0, 1.7, 0.001))
plt.show()
