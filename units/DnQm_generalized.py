import numpy as np
import matplotlib.pyplot as plt

n = 2
m = 8

if n == 2:
	direction_group_size = 4
	n_direction_groups = int(m/direction_group_size)

	direction_vectors = []
	direction_xs = []
	direction_ys = []
	for j in range(n_direction_groups):
		for k in range(direction_group_size):
			i = (j*4 + k)

			print(i, j**2 + 1, i - 4 * j, i * np.pi/2**(i + 2))

			theta = np.sqrt(j**2 + 1) * np.cos((i - 4 * j) * np.pi/2 + j * np.pi/2**(j + 2))

			x = np.cos(theta)
			y = np.sin(theta)

			direction_vector = [x, y]
			direction_vectors.append([x, y])
			direction_xs.append(x)
			direction_ys.append(y)
			# print(i + 1, direction_vector)
	
	fig, ax = plt.subplots()
	ax.scatter(direction_xs, direction_ys)
	plt.show()

