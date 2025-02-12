import matplotlib.pyplot as plt
import numpy as np

"""
Characteristics:
- D2Q8
- Multiple square sources
- Radiative scattering
- Run status: Not fully implemented - still mostly a copy of other scripts!
"""

def main():
    """ Lattice Boltzmann Simulation """
    
    # Simulation parameters
    Nx                     = 7 # resolution x-dir
    Ny                     = 7 # resolution y-dir
    dx                     = 1/(Nx-1)
    rho0                   = 100    # average density
    tau                    = 10     # collision timescale
    Nt                     = 4000   # number of timesteps
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
    # nu = 1 cm/s
    # Blue - Purely scattering (kappa=1/cm, j=0)
    # Red - Purely absorbing (kappa=10/cm, j=0/cm)
    # White - Source and scattering (kappa=1/cm, j=1/cm)
    F = np.zeros((Ny,Nx,NL)) #* rho0 / NL

    np.random.seed(42)
    F += 0.01*np.random.randn(Ny,Nx,NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    # F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))
    # rho = np.sum(F, 2)
    # for i in idxs:
    #     F[:,:,i] *= rho0 / rho
    
    # Procedurally-placed squares


    # Prep figure
    # fig = plt.figure(figsize=(4,2), dpi=200)
    fig, ax = plt.subplots(dpi=200)
    
    # Simulation Main Loop
    for it in range(Nt):
        F_old = np.array(F, copy=True)
        #print(it)
        
        # Set absorb boundaries
        # # x-left
        F[:, 0, [0, 4, 7]] = F[:, 1, [0, 4, 7]]
        # # x-right
        F[:, -1, [2, 5, 6]] = F[:, -2, [2, 5, 6]]
        # y-top
        F[0, :, [1, 4, 5]] = F[1, :, [1, 4, 5]]
        # y-bottom
        F[-1, :, [3, 6, 7]] = F[-2, :, [3, 6, 7]]
        
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
        
        # Calculate star boundaries
        starF_boundaries = []
        for starF in starFs:
            starF_boundary = F[starF,:]
            starF_boundary = 0.5
            starF_boundaries.append(starF_boundary)
        
        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux  = np.sum(F*cxs, 2) / rho
        uy  = np.sum(F*cys, 2) / rho
        
        # Apply Collision
        S = 10
        for i, w in zip(idxs, weights):
            F[:, :, i] = F[:, :, i] - dx * (F[:, :, i] - w * S)

        # Feq = np.zeros(F.shape)
        # for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        #     Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
        # 
        # F += -(1.0/tau) * (F - Feq)
        
        # Apply star boundaries
        for starF, starF_boundary in zip(starFs, starF_boundaries):
            F[starF,:] = starF_boundary
        
        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 5) == 0) or (it == Nt-1):
            # ux[cylinder] = 0
            # uy[cylinder] = 0
            # vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            # vorticity[cylinder] = np.nan
            # vorticity = np.ma.array(vorticity, mask=cylinder)
            
            # plt.imshow(vorticity, cmap='bwr')
            plt.imshow(np.sum(F, axis=2))
            # plt.imshow(~cylinder, cmap='gray', alpha=0.3)

            print(np.min(np.sum(F, axis=2)), np.max(np.sum(F, axis=2)))
           
            clim_magnitude = 1E1
            plt.clim(-clim_magnitude, +clim_magnitude+1)
            # ax = plt.gca()
            # ax.invert_yaxis()
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # ax.set_aspect('equal')

            if np.sum(F_old) > 0 and it < Nt - 1:
                error = np.abs(np.sum(F - F_old)/np.sum(F_old))
                print(f"error: {error}")
                
                if error < 1E-9:
                    break
                else:
                    plt.pause(0.0001)
                    plt.cla()
            
    
    # Save figure
    plt.savefig('latticeboltzmann.png', dpi=240)
    plt.show()
        
    return 0

if __name__== "__main__":
  main()
