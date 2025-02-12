import matplotlib.pyplot as plt
import numpy as np

"""
Characteristics:
- D2Q8
- Initial fluctuations
- Single cylindrical source
- BGK collisions
- Run status: Does not seem to work
"""

def main():
	# Simulation parameters
	L = 1
	Nx = 200
	Ny = 200
	dx = L/(Nx-1)
	rho0 = 1
	tau = 0.1
	Nt = 1000
	plotRealTime = True
	
	# Lattice speeds / weights
	NL = 8
	idxs = np.arange(NL)

	c0 = 1
	cx_verthoriz = [round(c0 * np.cos((i-1) * np.pi/2)) for i in range(1, 4+1)]
	cy_verthoriz = [round(c0 * np.sin((i-1) * np.pi/2)) for i in range(1, 4+1)]
	cx_diag = [round(c0 * np.sqrt(2) * np.cos((i-5) * np.pi/2 + np.pi/4)) for i in range(5, 8+1)]
	cy_diag = [round(c0 * np.sqrt(2) * np.sin((i-5) * np.pi/2 + np.pi/4)) for i in range(5, 8+1)]

	cxs = np.array([*cx_verthoriz, *cx_diag])
	cys = np.array([*cy_verthoriz, *cy_diag])

	w_verthoriz = 0.2
	w_diag = 0.05
	weights = np.array([w_verthoriz, w_verthoriz, w_verthoriz, w_verthoriz, w_diag, w_diag, w_diag, w_diag])
	assert np.sum(weights) == 1
	
	# Initial Conditions
	F = np.ones((Ny,Nx,NL)) #* rho0 / NL
	np.random.seed(42)
	F += 0.01*np.random.randn(Ny,Nx,NL)
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho

	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx/2)**2 + (Y - Ny/2)**2 < (Ny/16)**2

	kappa_a = 1E-1
	kappa_s = 1E-0
	beta = kappa_a + kappa_s
	s = np.zeros((Ny, Nx, NL))
	s[cylinder, :] = 1E1

	# Prep figure
	fig = plt.figure(figsize=(4, 4), dpi=80)
	
	# Simulation Main Loop
	for it in range(Nt):
		print(f"iteration {it:4d}")

		F_old = np.array(F, copy=True)

		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		# BGK collision
		rho = np.sum(F, 2)
		ux  = np.sum(F*cxs, 2) / rho
		uy  = np.sum(F*cys, 2) / rho
		Feq = np.zeros(F.shape)
		for i, cx, cy, w in zip(idxs, cxs, cys, weights):
			Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
			F += -(1.0/tau) * (F - Feq)
		
		# # # ASE effects
		# for i, w in zip(idxs, weights):
		# 	# new
		# 	# F[:, :, i] = F[:, :, i] - dx * (beta * F[:, :, i] - kappa_s * w * np.sum(F[:, :, i]) + kappa_a * w * s[:, :, i])
		# 	# old
		# 	F[:, :, i] = F[:, :, i] - dx * (F[:, :, i] - w * s[:, :, i])

		# Apply cylinder boundary
		F[cylinder,:] = 1E1
		
		# Set absorb boundaries
		F[:, 0, [0, 4, 7]] = F[:, 1, [0, 4, 7]]
		F[:, -1, [2, 5, 6]] = F[:, -2, [2, 5, 6]]
		F[0, :, [1, 4, 5]] = F[1, :, [1, 4, 5]]
		F[-1, :, [3, 6, 7]] = F[-2, :, [3, 6, 7]]
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
			plt.imshow(np.sum(F, axis=2), cmap='bwr')
			plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			plt.clim(-.1, .1)

			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			ax.set_aspect('equal')

			if np.sum(F_old) > 0 and it < Nt - 1:
				error = np.abs(np.sum(F - F_old)/np.sum(F_old))
				print(f"error: {error}")
				
				if error < 1E-13:
					break
				else:
					plt.pause(0.0001)
					plt.cla()
			
	
	# Save figure
	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()
		
	return 0



if __name__== "__main__":
  main()


