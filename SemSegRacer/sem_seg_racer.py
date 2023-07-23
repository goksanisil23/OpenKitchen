import glob
import os

import matplotlib.pyplot as plt
import numpy as np

TRACK_DATABASE_DIR = "/home/s0001734/Downloads/racetrack-database/tracks/"

all_track_files = glob.glob(os.path.join(TRACK_DATABASE_DIR, "*.csv"))

for track_file in all_track_files:
    track_data = np.loadtxt(
        track_file,
        comments="#",
        delimiter=",",
    )

    center_x = track_data[:, 0]
    center_y = track_data[:, 1]
    width_right = track_data[:, 2]
    width_left = track_data[:, 3]

    dx = np.gradient(center_x)
    dy = np.gradient(center_y)
    mag = np.sqrt(dx**2 + dy**2)
    heading_x = dx / mag
    heading_y = dy / mag

    right_bound_x = [
        c_x + d * del_y for c_x, d, del_y in zip(center_x, width_right, heading_y)
    ]
    right_bound_y = [
        c_y - d * del_x for c_y, d, del_x in zip(center_y, width_right, heading_x)
    ]

    left_bound_x = [
        c_x - d * del_y for c_x, d, del_y in zip(center_x, width_left, heading_y)
    ]
    left_bound_y = [
        c_y + d * del_x for c_y, d, del_x in zip(center_y, width_left, heading_x)
    ]

    plt.scatter(center_x[0], center_y[0], s=1000)
    plt.plot(center_x, center_y, marker="o", color="black")
    plt.plot(left_bound_x, left_bound_y, marker="o", color="blue")
    plt.plot(right_bound_x, right_bound_y, marker="o", color="red")
    for c_x, c_y, h_x, h_y in zip(center_x, center_y, heading_x, heading_y):
        plt.arrow(
            c_x,
            c_y,
            h_x,
            h_y,
            head_width=0.7,
            head_length=0.3,
            fc="red",
            ec="red",
            linewidth=3,
        )
        break

    plt.show()
