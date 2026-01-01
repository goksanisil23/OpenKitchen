# Note: The pybind so must be discoverable in PYTHONPATH

import open_kitchen_pybind as ok
import numpy as np
import matplotlib.pyplot as plt

from time import sleep


env = ok.Environment(
    "/home/s0001734/Downloads/racetrack-database/tracks/Montreal.csv",
    draw_rays=False,
    hidden_window=False,
)
# env.set_action(throttle, steering)
while True:
    env.step()
    env.set_action(30.0, 0.0)
    img = np.frombuffer(bytes(env.get_render_target()), dtype=np.uint8)
    info = env.get_render_target_info()
    img = np.flip(img.reshape(info.height, info.width, info.channels), axis=0)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
