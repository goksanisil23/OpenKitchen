import matplotlib.pyplot as plt
import numpy as np

track_data = np.loadtxt('/home/s0001734/Downloads/racetrack-database/tracks/Austin.csv', comments='#', delimiter=',')

center_x = track_data[:,0]
center_y = track_data[:,1]
width_right = track_data[:,2]
width_left = track_data[:,3]

dx = np.gradient(center_x)
dy = np.gradient(center_y)
mag = np.sqrt(dx**2+dy**2)
heading_x = dx / mag
heading_y = dy / mag

left_bound_x = [x_i - width_i*dy_i for x_i, width_i, dy_i in zip(center_x,width_left,heading_x)]
right_bound_x = [x_i + width_i*dy_i for x_i, width_i, dy_i in zip(center_x,width_right,heading_x)]

left_bound_y = [y_i + width_i*dx_i for y_i, width_i, dx_i in zip(center_y,width_left,heading_y)]
right_bound_y = [y_i - width_i*dx_i for y_i, width_i, dx_i in zip(center_y,width_left,heading_y)]

plt.scatter(center_x[0],center_y[0],s=1000)
plt.plot(center_x, center_y, marker="o",color="black") 
plt.scatter(left_bound_x, left_bound_y)
plt.scatter(right_bound_x, right_bound_y)





plt.show()
