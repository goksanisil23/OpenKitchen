import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def plot_histogram_from_files(folder_path, file_extension="*.txt"):
    search_path = os.path.join(folder_path, file_extension)
    files = glob.glob(search_path)

    steering_vals = []

    print(f"Found {len(files)} files. Processing...")

    # 2. Loop through files and extract data
    # for file_path in files:
    for file_path in tqdm.tqdm(files):
        with open(file_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            # val = float(parts[1])
            val = float(parts[0])
            steering_vals.append(val)

    # hist, bin_edges = np.histogram(steering_vals, binds=50)
    plt.hist(steering_vals, bins=50)
    plt.show()


plot_histogram_from_files("build/SaoPaulo")
