import matplotlib.pyplot as plt
import numpy as np

"""
Characteristics:
- D2Q8 or D2Q16
- Initial fluctuations
- Multiple star sources
- Radiative scattering
- Run status: Better (but still low) angular resolution radiation, currently WIP (see redistribution_method notebook)
"""

def main(
		NL=16,
		Nx=800,
		Ny=800,
		rho0=1,
		Nt=500,
		c0=1,
		plotRealTime=True,
		analysis_output=False,
		method="RTE"
	):
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters
	dx = 1/(Nx-1)
	# (see function params)

	# Direction speeds/vectors
	idxs = np.arange(NL)

	if NL == 8:
		idxs_dir = [0, 4, 1, 5, 2, 6, 3, 7]
	elif NL == 16:
		idxs_dir = [0, 8, 4, 9, 1, 10, 5, 11, 2, 12, 6, 13, 3, 14, 7, 15]

	# D2Q4
	cx_verthoriz = [round(c0 * np.cos((i-1) * np.pi/2)) for i in range(1, 4+1)]
	cy_verthoriz = [round(c0 * np.sin((i-1) * np.pi/2)) for i in range(1, 4+1)]

	# D2Q8
	cx_diag = [round(c0 * np.sqrt(2) * np.cos((i-5) * np.pi/2 + np.pi/4)) for i in range(5, 8+1)]
	cy_diag = [round(c0 * np.sqrt(2) * np.sin((i-5) * np.pi/2 + np.pi/4)) for i in range(5, 8+1)]

	if NL == 16:
		# D2Q16
		cx_diag2 = [round(c0 * np.sqrt(5) * np.cos((i-9) * np.pi/4 + np.pi/8)) for i in range(9, 16+1)]
		cy_diag2 = [round(c0 * np.sqrt(5) * np.sin((i-9) * np.pi/4 + np.pi/8)) for i in range(9, 16+1)]

	# D2Q4
	cxs = np.array(cx_verthoriz)
	cys = np.array(cy_verthoriz)

	# D2Q8
	cxs = np.append(cxs, cx_diag)
	cys = np.append(cys, cy_diag)

	if NL == 16:
		# D2Q16
		cxs = np.append(cxs, cx_diag2)
		cys = np.append(cys, cy_diag2)

	# Direction weights
	if NL == 8:
		# D2Q8
		w_verthoriz = 0.2
		w_diag = 0.05
		weights = np.array([[w_verthoriz] * 4, [w_diag] * 4]).flatten()
	elif NL == 16:
		# D2Q16
		w_verthoriz = 0.196
		w_diag = 0.028
		w_diag2 = 0.013
		weights = np.array([*([w_verthoriz] * 4), *([w_diag] * 4), *([w_diag2] * 8)]).flatten()
		# print(weights, np.sum(weights))

	assert np.sum(weights) == 1

	# Initial Conditions
	F = np.zeros((Ny,Nx,NL)) #* rho0 / NL
	np.random.seed(42)
	F += 0.01 * np.random.randn(Ny, Nx, NL)
	# rho = np.sum(F, 2)
	# for i in idxs:
	# 	F[:,:,i] *= rho0 / rho

	# Coordinate mesh
	X, Y = np.meshgrid(range(Nx), range(Ny))

	# # Procedurally-placed stars
	# nx = 1
	# ny = 1
	# star_centers = []
	# for i in range(1, nx+1):
	# 	for j in range(1, ny+1):
	# 		star_centers.append([Nx/(nx+1) * i, Ny/(ny+1) * j])

	# Hard-coded stars
	# star_centers = [[Ny, 0]]
	star_centers = [[Ny//2, Nx//2]]
	
	starFs = []
	for center in star_centers:
		starF = (X - center[0])**2 + (Y - center[1])**2 < 1**2
		starFs.append(starF)

	# Prep figure
	fig, ax = plt.subplots(dpi=200)

	# print(idxs_dir)
	# print(cxs)
	# print(cys)

	# Simulation Main Loop
	for it in range(Nt):
		F_old = np.array(F, copy=True)
		#print(it)

		# Drift
		for i, i_dir in zip(idxs, idxs_dir):
			# i_next = (i + 1) % NL
			# i_prev = (i - 1) % NL
			# i_dir_next = idxs_dir[i_next]
			# i_dir_prev = idxs_dir[i_prev]

			# print(i_prev, i, i_next, i_dir_prev, i_dir, i_dir_next)

			cx = cxs[i_dir]
			cy = cys[i_dir]
			# cx_next = cxs[i_dir_next]
			# cx_prev = cxs[i_dir_prev]
			# cy_next = cys[i_dir_next]
			# cy_prev = cys[i_dir_prev]

			# print(f"({cx_prev},{cy_prev}), ({cx},{cy}), ({cx_next},{cy_next})")

			# Smearing fractions
			smear_carrier = 1.00#0.5
			smear_sidebands = (1-smear_carrier)/2

			# carrier direction
			F[:,:,i_dir] = np.roll(smear_carrier * F[:,:,i_dir], cx, axis=1)
			F[:,:,i_dir] = np.roll(smear_carrier * F[:,:,i_dir], cy, axis=0)

			# # +1 sideband direction
			# F[:,:,i_dir_next] = np.roll(smear_sidebands * F[:,:,i_dir], cx, axis=1)
			# F[:,:,i_dir_next] = np.roll(smear_sidebands * F[:,:,i_dir], cy, axis=0)
			#
			# # -1 sideband direction
			# F[:,:,i_dir_prev] = np.roll(smear_sidebands * F[:,:,i_dir], cx, axis=1)
			# F[:,:,i_dir_prev] = np.roll(smear_sidebands * F[:,:,i_dir], cy, axis=0)
		
		# Set absorb boundaries
		if NL == 8:
			# x-left
			F[:, 0, [0, 4, 7]] = F[:, 1, [0, 4, 7]]
			# x-right
			F[:, -1, [2, 5, 6]] = F[:, -2, [2, 5, 6]]
			# y-top
			F[0, :, [1, 4, 5]] = F[1, :, [1, 4, 5]]
			# y-bottom
			F[-1, :, [3, 6, 7]] = F[-2, :, [3, 6, 7]]
		elif NL == 16:
			# x-left
			F[:, :2, [0, 4, 7, 8, 15]] = 0#F[:, 1, [0, 4, 7, 8, 15]]
			# x-right
			F[:, -2:, [2, 5, 6, 11, 12]] = 0#F[:, -2, [2, 5, 6, 11, 12]]
			# y-top
			F[:2, :, [1, 4, 5, 9, 10]] = 0#F[1, :, [1, 4, 5, 9, 10]]
			# y-bottom
			F[-2:, :, [3, 6, 7, 13, 14]] = 0#F[-2, :, [3, 6, 7, 13, 14]]
		
		# Calculate star boundaries
		starF_boundaries = []
		for starF in starFs:
			starF_boundary = F[starF,:]
			starF_boundary = 1
			starF_boundaries.append(starF_boundary)
		
		if method == "RTE":
			# Apply scattering
			S = 0E+0
			for i, w in zip(idxs, weights):
				print(np.sum(F[:, :, i]), dx, dx * np.sum(F[:, :, i]))
				F[:, :, i] = F[:, :, i] - dx * (F[:, :, i] - w * S)
		elif method == "BGK":
			# Calculate fluid variables
			rho = np.sum(F, 2)
			ux  = np.sum(F*cxs, 2) / rho
			uy  = np.sum(F*cys, 2) / rho
			tau = 0.6

			Feq = np.zeros(F.shape)
			for i, cx, cy, w in zip(idxs, cxs, cys, weights):
				Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )

			F += -(1.0/tau) * (F - Feq)
		
		# Apply star boundaries
		for starF, starF_boundary in zip(starFs, starF_boundaries):
			F[starF,:] = starF_boundary
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 5) == 0) or (it == Nt-1):
			plt.imshow(np.sum(F, axis=2))
			# plt.imshow(~cylinder, cmap='gray', alpha=0.3)

			# print(np.min(np.sum(F, axis=2)), np.max(np.sum(F, axis=2)))

			# clim_magnitude = 1E1
			# plt.clim(-clim_magnitude, +clim_magnitude+1)
			# ax = plt.gca()
			# ax.invert_yaxis()
			# ax.get_xaxis().set_visible(False)
			# ax.get_yaxis().set_visible(False)
			# ax.set_aspect('equal')

			if np.sum(F_old) > 0 and it < Nt - 1:
				error = np.abs(np.sum(F - F_old)/np.sum(F_old))
				print(f"error: {error}")
				
				if error < 1E-9:
					continue
					# break
				else:
					plt.pause(0.0001)
					plt.cla()
	
	print(F.shape)
	print(np.sum(F[Ny//2, Nx//2:, :], axis=1))
			
	# Save formatted figure
	# plt.cla()
	#
	# fig, ax = plt.subplots()
	#
	# F_final = np.sum(F, axis=2)
	#
	# plot = plt.imshow(F_final/np.max(F_final))
	#
	# ax.set_xlabel("X")
	# ax.set_ylabel("Y")
	#
	# # plt.colorbar(label="Normalized intensity")
	#
	# from mpl_toolkits.axes_grid1 import make_axes_locatable
	# divider = make_axes_locatable(ax)
	# cbaxes = divider.append_axes("right", size="5%", pad=0.0)  # Adjust pad to 0 for no gap
	# cb = fig.colorbar(plot, cax=cbaxes, label="Normalized intensity")
	#
	# ax.set_aspect(1)
	# plt.tight_layout()
	# plt.gcf().set_dpi(500)
	#
	# plt.savefig(f'latticeboltzmann_{Nx}x{Ny}.png', dpi=500)
	# plt.show()

	if analysis_output:
		# Calculate anisotropy
		slice_r = []
		slice_I = []

		hx = star_centers[0][0]
		hy = star_centers[0][1]
		hz = 0

		k = 0 # we select the slice z = 0
		for i in range(Nx):
			for j in range(Ny):
				r = np.sqrt((i-hx)**2 + (hy-j-1)**2 + (k-hz)**2)
				slice_r.append(r/(Nx))
				slice_I.append(np.sum(F[j, i]))
				
				# print(r, (x, y, z), intensity_slice[x, y].value)

		slice_I /= np.max(slice_I)

		slice_r_groups = {}

		for r, I in zip(slice_r, slice_I):
			if r in slice_r_groups:
				slice_r_groups[r].append(I)
			else:
				slice_r_groups[r] = [I]

		slice_r_values = []
		slice_I_means = []
		slice_I_stdevs = []
		for r, I_list in slice_r_groups.items():
			slice_r_values.append(r)
			slice_I_means.append(np.mean(I_list))
			slice_I_stdevs.append(np.std(I_list))

		slice_I_means = np.array(slice_I_means)
		slice_I_stdevs = np.array(slice_I_stdevs)

		print(np.shape(slice_r_values), np.shape(slice_I_means), np.shape(slice_I_stdevs))
		print(np.std(slice_I), np.mean(slice_I_stdevs))

		fig, ax = plt.subplots(figsize=(8,6))

		plt.scatter(slice_r_values, slice_I_stdevs, s=1)
		# plt.scatter(slice_r_values, slice_I_means, s=1)
		# plt.scatter(slice_r_values, slice_I_stdevs/slice_I_means, s=1)
		# plt.axhline(y=3E1, c="red", linestyle="--")

		# ax.set_xscale("log")
		# ax.set_yscale("log")

		plt.title("Plot of anisotropy vs. normalized radius")
		ax.set_xlabel("Normalized radius")
		ax.set_ylabel("Anisotropy ($\\sigma_I$)")

		# ax.set_aspect(1)
		plt.gcf().set_dpi(500)
		plt.tight_layout()

		plt.xlim(0, 0.3)

		plt.savefig("redistribution_anisotropy.png", dpi=1000)
		plt.show()

	return 0

if __name__== "__main__":
  main(NL=8, Nx=100, Ny=100, plotRealTime=False, analysis_output=False)

