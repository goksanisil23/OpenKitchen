import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Load the model
model = torch.jit.load("q_network_deep_sarsa.pth")
print(model)
model.eval()

# Create a 10x10 grid and predict Q-values
grid_size = 10
q_values_map = np.zeros((grid_size, grid_size, 4))  # 4 for the number of actions

for i in range(grid_size):
    for j in range(grid_size):
        state = torch.tensor([i, j], dtype=torch.float32)
        with torch.no_grad():
            q_values = model(state)
            # q_values = model(state).numpy()
        # q_values_map[i, j] = q_values

# # Visualization
# fig, ax = plt.subplots(figsize=(15, 15))
# for i in range(grid_size):
#     for j in range(grid_size):
#         ax.text(
#             j,
#             grid_size - 1 - i,
#             f"U:{q_values_map[i, j, 0]:.2f}",
#             va="top",
#             ha="left",
#             color="blue",
#         )
#         ax.text(
#             j,
#             grid_size - 1 - i,
#             f"D:{q_values_map[i, j, 1]:.2f}",
#             va="bottom",
#             ha="left",
#             color="red",
#         )
#         ax.text(
#             j,
#             grid_size - 1 - i,
#             f"L:{q_values_map[i, j, 2]:.2f}",
#             va="bottom",
#             ha="right",
#             color="green",
#         )
#         ax.text(
#             j,
#             grid_size - 1 - i,
#             f"R:{q_values_map[i, j, 3]:.2f}",
#             va="top",
#             ha="right",
#             color="purple",
#         )

# ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
# ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
# ax.grid(which="minor", color="gray", linestyle="-", linewidth=2)
# ax.tick_params(which="minor", size=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().invert_yaxis()  # to match the (0,0) with the bottom-left corner
# plt.show()
