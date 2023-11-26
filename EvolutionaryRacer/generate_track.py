import matplotlib.pyplot as plt
import numpy as np

# Step 1: Generate a set of track center coordinates (x, y)
# Here I am creating a simple oval track as an example
theta = np.linspace(0, 2 * np.pi, 100)
x_center = 100 * np.cos(theta)
y_center = 50 * np.sin(theta)

# Step 2: Define the width of the track at each point
# Here I'm using a constant width, but you could vary the width at different points if you like
track_width = 5

# Step 3: Calculate the left and right boundaries of the track
# We calculate the perpendicular offset at each point to find the boundary points
dx = np.gradient(x_center)
dy = np.gradient(y_center)
norm = np.sqrt(dx**2 + dy**2)
x_left = x_center - track_width / 2 * dy / norm
y_left = y_center + track_width / 2 * dx / norm
x_right = x_center + track_width / 2 * dy / norm
y_right = y_center - track_width / 2 * dx / norm


# Step 4: Save the track data to a file
track_data = np.vstack(
    (
        x_center,
        y_center,
        np.full_like(x_center, track_width),
        np.full_like(x_center, track_width),
    )
).T
np.savetxt(
    "oval_track.txt",
    track_data,
    header="track_center_x,track_center_y,track_width_right,track_width_left",
    delimiter=",",
    comments="",
)


# Step 4: Plot the track
plt.figure()
plt.plot(x_center, y_center, "k-", label="Centerline")
plt.plot(x_left, y_left, "r-", label="Left boundary")
plt.plot(x_right, y_right, "b-", label="Right boundary")
# plt.fill_betweenx(y_center, x_left, x_right, color="gray", alpha=0.5)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.grid(True)
plt.show()
