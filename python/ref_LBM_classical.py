import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid

"""



def main():
	""" Lattice Boltzmann Simulation """
	
	# Simulation parameters
	L = 10# optical thickness
	Nx = 400# resolution x-dir
	Ny = 400# resolution y-dir
	dx = L/(Nx-1)
	rho0 = 100# average density
	tau = 0.6# collision timescale
	Nt = 4000   # number of timesteps
	plotRealTime = True # switch on for plotting as the simulation goes along
	
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
	X, Y = np.meshgrid(range(Nx), range(Ny))
	rho = np.sum(F,2)
	for i in idxs:
		F[:,:,i] *= rho0 / rho
	
	# Cylinder boundary
	X, Y = np.meshgrid(range(Nx), range(Ny))
	cylinder = (X - Nx/2)**2 + (Y - Ny/2)**2 < (Ny/64)**2
	
	# Prep figure
	fig = plt.figure(figsize=(4, 4), dpi=80)
	
	# Simulation Main Loop
	for it in range(Nt):
		print(it)

		print(np.min(F), np.max(F))
		
		# Drift
		for i, cx, cy in zip(idxs, cxs, cys):
			F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
			F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
		
		# Set reflective boundaries
		# bndryF = F[cylinder,:]
		# bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]
	
		# Calculate fluid variables
		rho = np.sum(F,2)
		ux  = np.sum(F*cxs,2) / rho
		uy  = np.sum(F*cys,2) / rho
		
		# Apply Collision
		# Feq = np.zeros(F.shape)
		# for i, cx, cy, w in zip(idxs, cxs, cys, weights):
		# 	Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
		# F += -(1.0/tau) * (F - Feq)
		
		# Apply Collision
		S = 1E1
		for i, w in zip(idxs, weights):
			F[:, :, i] = F[:, :, i] - dx * (F[:, :, i] - w * S)

		# Apply cylinder boundary
		F[cylinder,:] = 1E1#bndryF
		
		# Set absorb boundaries
		F[:, 0, [0, 4, 7]] = F[:, 1, [0, 4, 7]]
		F[:, -1, [2, 5, 6]] = F[:, -2, [2, 5, 6]]
		F[0, :, [1, 4, 5]] = F[1, :, [1, 4, 5]]
		F[-1, :, [3, 6, 7]] = F[-2, :, [3, 6, 7]]
		
		
		# plot in real time - color 1/2 particles blue, other half red
		if (plotRealTime and (it % 10) == 0) or (it == Nt-1):
			plt.cla()
			# ux[cylinder] = 0
			# uy[cylinder] = 0
			# vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
			# vorticity[cylinder] = np.nan
			# vorticity = np.ma.array(vorticity, mask=cylinder)
			plt.imshow(np.sum(F, axis=2), cmap='bwr')
			plt.imshow(~cylinder, cmap='gray', alpha=0.3)
			plt.clim(-.1, .1)
			ax = plt.gca()
			ax.invert_yaxis()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)	
			ax.set_aspect('equal')	
			plt.pause(0.001)
			
	
	# Save figure
	plt.savefig('latticeboltzmann.png',dpi=240)
	plt.show()
		
	return 0



if __name__== "__main__":
  main()


