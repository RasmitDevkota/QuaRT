import sys

import numpy as np
np.set_printoptions(linewidth=150)

import matplotlib.pyplot as plt

def prints(x):
	print(np.sum(x, axis=2))

def printd(s):
	if "d" in sys.argv:
		print(s)

k = 100//2
N = 2*k + 1
NL = 8

idxs_dir = [0, 4, 1, 5, 2, 6, 3, 7]
cxs_unsorted = [1, 0, -1,  0, 1, -1, -1,  1]
cys_unsorted = [0, 1,  0, -1, 1,  1, -1, -1]
cxs = cxs_unsorted
cys = cys_unsorted
# cxs = [cxs_unsorted[i_dir] for i_dir in idxs_dir]
# cys = [cys_unsorted[i_dir] for i_dir in idxs_dir]

# smearing fractions
smear_carrier = 1.0
smear_sidebands = (1-smear_carrier)/2

# initial setup
x0 = np.zeros(shape=(N, N, NL))

source_positions = [
	(0+1, N-1-1)
]

# x0[0, 0, :] = # bottom left
# x0[0, N-1, :] = 2 # bottom right
# x0[N-1, 0, :] = 3 # top left
# x0[N-1, N-1, :] = 4 # top right

prints(x0[::-1, :])

print("-----")

def emit(xi):
	xf = np.array(xi, copy=True)

	# Source boundary conditions
	for source_position in source_positions:
		source_x, source_y = source_position
		xf[source_y, source_x, :] = 1

	return xf

def absorb(xi):
	xf = np.array(xi, copy=True)

	# # x=0, y=0
	# xf[0, 0, :] = 0
	#
	# # x=x, y=0
	# xf[0, :, :] = 0
	#
	# # x=0, y=y
	# xf[:, 0, :] = 0
	#
	# # x=x, y=y
	# xf[:, :, :] = 0

	return xf

def propagate(xi):
	xf = np.array(xi, copy=True)

	for dir_i, i_dir in enumerate(idxs_dir):
		nl = i_dir
		cy = cys[i_dir]
		cx = cxs[i_dir]

		for j in range(N):
			for i in range(N):
				l_src = xi[j, i, nl]

				dir_i_next = (dir_i+1) % NL
				dir_i_prev = (dir_i-1) % NL
				nl_next = idxs_dir[dir_i_next]
				nl_prev = idxs_dir[dir_i_prev]

				cx_next = cxs[nl_next]
				cy_next = cys[nl_next]
				cx_prev = cxs[nl_prev]
				cy_prev = cys[nl_prev]

				printd(l_src)

				# stream the value to the next lattice in the same direction
				printd(f"stream - xf[{(j + cy) % N, (i + cx) % N, nl}] = {xf[(j + cy) % N, (i + cx) % N, nl]} + {smear_carrier * l_src}")
				xf[(j + cy) % N, (i + cx) % N, nl] += smear_carrier * l_src

				# if np.sum(l_src) > 0 or True:
				# 	print(f"lattice indices: {nl_prev} {nl} {nl_next}")
				# 	print(f"lattice directions: {cx_prev, cy_prev} {cx, cy} {cx_next, cy_next}")
				# 	print(f"lattice directions: {cy_prev, cx_prev} {cy, cx} {cy_next, cx_next}")
				# 	print(f"this lattice point: {j, i}, {nl}")
				# 	print(f"the carrier lattice dest: {(j + cy) % N, (i + cx) % N}, {nl}")
				# 	print(f"the next lattice dest: {(j + cy_next) % N, (i + cx_next) % N}, {nl_next}")
				# 	print(f"the prev lattice dest: {(j + cy_prev) % N, (i + cx_prev) % N}, {nl_prev}")
				# 	print("transffering the following amount to the carrier lattice dest:", smear_carrier * l_src, xf[(j + cy) % N, (i + cx) % N, nl])
				# 	print("transffering amount to the next lattice dest:", (1-smear_carrier) * l_src, xf[(j + cy_next) % N, (i + cx_next) % N, nl_next])
				# 	print("transffering amount to the prev lattice dest:", (1-smear_carrier) * l_src, xf[(j + cy_prev) % N, (i + cx_prev) % N, nl_prev])
				#
				# 	print("-----")

				# redistribute the value to the next lattice in the adjacent directions
				printd(f"smear - xf[{(j + cy_next) % N, (i + cx_next) % N, nl_next}] = {xf[(j + cy_next) % N, (i + cx_next) % N, nl_next]} + {(1-smear_carrier)/2 * l_src}")
				printd(f"smear - xf[{(j + cy_prev) % N, (i + cx_prev) % N, nl_prev}] = {xf[(j + cy_prev) % N, (i + cx_prev) % N, nl_prev]} + {(1-smear_carrier)/2 * l_src}")
				xf[(j + cy_next) % N, (i + cx_next) % N, nl_next] += (1-smear_carrier)/2 * l_src
				xf[(j + cy_prev) % N, (i + cx_prev) % N, nl_prev] += (1-smear_carrier)/2 * l_src

		# break

	# conserve total by decrementing the value at the source
	# xf -= xi
	# anomaly = np.sum(xf) - np.sum(xi)
	# if anomaly != 0.0:
	# 	print("Total distribution not conserved!")
	# 	# prints(xf[::-1, :])
	# 	print(anomaly)
	# 	return None

	return xf

n_it = 100
x_prev = np.array(x0, copy=True)
for it in range(n_it):
	print(f"iteration {it+1} ---")

	# copy the previous array
	x = np.array(x_prev, copy=True)

	# enforce boundary conditions - sources
	x = emit(x)

	# enforce boundary conditions - boundaries
	x = absorb(x)

	# propagate - streaming + smearing
	x = propagate(x)

	if x is None:
		break

	# print(x[:, :, 0][::-1, :])
	# prints(x[::-1, :])

	plt.imshow(np.sum(x, axis=2))
	if it < n_it - 1:
		plt.pause(1E-4)
		plt.cla()

	x_prev = x
print("----------")

