import numpy as np
import matplotlib.pyplot as plt

# Dimension scales
t_0 = 1 # yr
L_0 = 1 # pc

# Initial conditions
T = 1E4
n_H = 1E-3
R_s = 5.4E3
t_rec = 122.4E6

I_0 = 5E48 * 13.6 * 1.6E-19

# Simulation parameters
n_it = 50
Delta_t = 1E5
N = int(6.6E3)

# Problem setup
fig, ax = plt.subplots()

# Main loop
grid = np.zeros((N, N))

X, Y = np.meshgrid(range(N), range(N))
# source = (X)**2 + (Y - N)**2 < min(N//100, 10)**2
# grid[source] = 0.1#0.001

t_list = []
r_list = []

for it in range(n_it):
	print(f"it {it+1}")

	t = it * Delta_t
	print(-t/t_rec, np.exp(-t/t_rec))
	r = R_s * (1 - np.exp(-t/t_rec))**(1/3)

	t_list.append(t)
	r_list.append(r)

	Stromgen_sphere = (X)**2 + (Y - N)**2 < r**2
	grid[Stromgen_sphere] = 1

	# plt.imshow(grid)
	# if it < n_it - 1:
	# 	plt.pause(1E-5)
	# 	plt.cla()

data = np.array([t_list, r_list])
print(data)

plt.cla()

plt.scatter(t_list, r_list)

plt.show()

