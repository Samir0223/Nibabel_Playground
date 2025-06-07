import numpy as np
import matplotlib.pyplot as plt


A= np.array([[1, 2, 3], [4, 5, 6]])
# B= np.array([[7, 8], [9, 10], [11, 12]])
# C = A @ B
# print(C)

theta = np.pi/4
A_theta = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta), np.cos(theta)]])
B = np.array([[2,0], [0,2]])
xy = np.array([2, 3])
xy_theta = A_theta @ B @ xy
fig, ax = plt.subplots()
ax.scatter(xy[0], xy[1])
ax.plot([0, xy[0]], [0, xy[1]])
ax.scatter(xy_theta[0], xy_theta[1], marker='s')
ax.plot([0, xy_theta[0]], [0, xy_theta[1]], '--')
p = ax.axis("equal")


A_total = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
delta_x = 128
delta_y = 127
delta_z = 85.5
A_affine = np.zeros((4, 4))
A_affine[:3, :3] = A_total
A_affine[:, 3] = np.array([delta_x, delta_y, delta_z, 1])
origin = np.array([0, 0, 0, 1])
result = A_affine @ origin
# print(result)
B_inv = np.linalg.inv(B)
print(B_inv)
