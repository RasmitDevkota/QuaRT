import sys

import numpy as np
np.set_printoptions(linewidth=np.inf)

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 500

import matplotlib.pyplot as plt
plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "serif"
})

from matplotlib.colors import LogNorm

from qlbm_rt import simulate
from qlbm_utils import lattice_to_vector

UNUSED = None

# Fitting functions
fit_inv_sq = lambda x, k : k * np.power(x, -2)
fit_inv_sq_bg = lambda x, k, y_0 : y_0 + k * np.power(x, -2)

# Point source 1D test
def test_point_source_1D(N=None, n_timesteps=10):
    n = 1
    m = 2

    if N is None:
        N = [8]
    else:
        N = [int(N)]

    n_timesteps = int(n_timesteps)

    M_0 = np.prod(N)
    M = 2**int(np.ceil(np.log2(M_0)))

    lattice_I = np.zeros(shape=(*N, m))
    I_i = lattice_to_vector(lattice_I)

    lattice_S = np.zeros(shape=(*N, m))
    lattice_S[N[0]//2, :] = 1
    S_i = lattice_to_vector(lattice_S)

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        save_lattices=True,
        save_circuit=True
    )

    lattices = np.save(f"outputs/lattice_point_src_1D_{n_timesteps}.npy", lattices_I)

    analysis_point_source_1D(n_timesteps=n_timesteps, sep_dirs=True)

def analysis_point_source_1D(N=UNUSED, n_timesteps=10, sep_dirs=True, filename=False, show=False):
    if isinstance(sep_dirs, str):
        if sep_dirs.lower() in ["true", "yes", "ye", "y", "1"]:
            sep_dirs = True
        else:
            sep_dirs = False

    n_timesteps = int(n_timesteps)

    if filename is None:
        filename = f"outputs/lattice_point_src_1D_{n_timesteps}.npy"

    lattices = np.load(filename)

    fig, ax = plt.subplots(nrows=n_timesteps+1)

    for timestep, lattice in enumerate(lattices):
        if sep_dirs:
            ax[timestep].scatter(np.arange(lattice.shape[0]), lattice[:, 0][::-1], c="red", marker=">")
            ax[timestep].scatter(np.arange(lattice.shape[0]), lattice[:, 1][::-1], c="blue", marker="<")
        else:
            ax[timestep].scatter(np.arange(lattice.shape[0]), np.sum(lattice, axis=1)[::-1])

        ax[timestep].set_title(f"Timestep {timestep}")

    plot_filename = f"outputs/lattice_point_src_1D_{n_timesteps}.png"
    plt.savefig(plot_filename, dpi=500)
    print(f"Saved plot to {plot_filename}")

    if show:
        plt.show()

# Unidirectional radiation test
def test_unidirectional_source(n=2, m=8, N=None, n_timesteps=3):
    n = int(n)
    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    M = np.prod(N)

    # @TODO - apply multiprocessing
    lattices_I = [] # shape (mu, n_timesteps+1, *N, m)
    for mu in range(m):
        I_i = np.zeros(shape=(M, m))

        lattice_S = np.zeros(shape=(*N, m))
        lattice_S[*[Ni//2 for Ni in N], mu] = 1
        lattice_S = np.transpose(lattice_S, (1, 0, 2))
        S_i = lattice_to_vector(lattice_S)

        lattices_I_mu, lattices_S_mu, norms = simulate(
            I_i, S_i,
            n, m,
            N,
            n_timesteps=n_timesteps,
            save_lattices=True
        )

        lattices_I.append(lattices_I_mu)

    np.save(f"outputs/lattice_uni_src_{n_timesteps}_{m}.npy", lattices_I)

    analysis_unidirectional_source(m=m, n_timesteps=n_timesteps)

def analysis_unidirectional_source(n=UNUSED, m=8, N=UNUSED, n_timesteps=3, show=False):
    m = int(m)
    n_timesteps = int(n_timesteps)

    fig, ax = plt.subplots(figsize=(6,18), nrows=m, ncols=n_timesteps+1)

    lattices = np.load(f"outputs/lattice_uni_src_{n_timesteps}_{m}.npy", allow_pickle=True)
    N = lattices.shape[2:-1]
    for mu in range(m):
        print("/"*20 + " " + str(mu) + " " + "/"*20)
        lattices_mu = lattices[mu]
        for timestep, lattice in enumerate(lattices_mu):
            spwn = lattice[:,:,mu]
            sumspwn = np.sum(spwn)
            sums = np.sum(lattice)
            lttc = np.sum(np.sum(lattice, axis=0), axis=0)
            print(timestep, sumspwn, sums, "-", lttc)

            lattice[*[Ni//2 for Ni in N] ,:] = 0.25 * np.sum(lattice)
            ax[mu][timestep].imshow(lattice.sum(axis=2))

            ax[mu][timestep].set_title(f"Direction {mu} - Timestep {timestep}")

    plt.savefig(f"outputs/plot_uni_src_{n_timesteps}_{m}.png", dpi=500)

    if show:
        plt.show()

# Angular redistribution spread test
def test_redistribution_spread(n=2, m=8, mu_0=0, alpha=0.33, N=None, n_timesteps=3):
    n = int(n)
    m = int(m)

    mu_0 = int(mu_0)
    alpha = float(alpha)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    M = np.prod(N)

    angular_redistribution_coefficients = np.zeros((m,))
    angular_redistribution_coefficients[0] = alpha
    angular_redistribution_coefficients[+1] = (1-alpha)/2
    angular_redistribution_coefficients[-1] = (1-alpha)/2

    I_i = np.zeros(shape=(M, m))

    lattice_S = np.zeros(shape=(*N, m))
    lattice_S[*[Ni//2 for Ni in N], mu_0] = 1
    S_i = lattice_to_vector(lattice_S)

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        angular_redistribution_coefficients=angular_redistribution_coefficients,
        save_lattices=True, save_name="redist_spread"
    )

    np.save(f"outputs/lattice_redist_spread_{n_timesteps}_{m}_{mu_0}.npy", lattices_I)

    analysis_redistribution_spread(m=m, mu_0=mu_0, n_timesteps=n_timesteps)

def analysis_redistribution_spread(n=UNUSED, m=8, mu_0=0, alpha=UNUSED, N=UNUSED, n_timesteps=3, filename=None, show=False):
    n_timesteps = int(n_timesteps)

    if filename is None:
        filename = f"outputs/lattice_redist_spread_{n_timesteps}_{m}_{mu_0}.npy"

    lattices = np.load(filename)

    fig, ax = plt.subplots(ncols=n_timesteps+1)

    for timestep, lattice in enumerate(lattices):
        ax[timestep].imshow(lattice.sum(axis=2))

        ax[timestep].set_title(f"Timestep {timestep}")

    if show:
        plt.show()

# Isotropic source test
def test_isotropic_source(n=2, m=8, N=None, n_timesteps=3, source_location=None, renormalize=False, lattice_I=None, lattice_S=None):
    n = int(n)
    if n == 1:
        raise ValueError("Isotropic source test can only be run in 2 or 3 dimensions; for 1 dimension, consider the point_source_1D test")

    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    renormalize = isinstance(renormalize, str) and renormalize.lower() in ["true", "1"]

    M = np.prod(N)

    if lattice_I is None:
        lattice_I = np.zeros(shape=(*N, m))
        I_i = lattice_to_vector(lattice_I)
    else:
        if isinstance(lattice_I, str):
            lattice_I = np.load(lattice_I)
            lattice_I = lattice_I[-1]
        elif hasattr(lattice_I, "__iter__"):
            if np.shape(lattice_I) == (*N, m):
                lattice_I = lattice_I
            elif np.shape(lattice_I)[1:] == (*N, m):
                lattice_I = lattice_I[-1]
            else:
                raise ValueError(f"Could not resolve warm start lattice with shape {np.shape(lattice_I)}; expected (*N, m) or (*, *N, m)")
        else:
            raise TypeError()

        I_i = lattice_to_vector(lattice_I)

    lattice_S = np.zeros(shape=(*N, m))

    if source_location is None or source_location.lower() == "central":
        # Central source
        if renormalize:
            if n == 2:
                if m == 4:
                    lattice_S[*[N_i//2 for N_i in N], 0:4] = 1.0
                elif m == 8:
                    lattice_S[*[N_i//2 for N_i in N], 0:4] = 1.0
                    lattice_S[*[N_i//2 for N_i in N], 4:8] = 1.0/np.sqrt(2)
                elif m == 16:
                    lattice_S[*[N_i//2 for N_i in N], 0:4] = 1.0
                    lattice_S[*[N_i//2 for N_i in N], 4:8] = 1.0/np.sqrt(2)
                    lattice_S[*[N_i//2 for N_i in N], 4:8] = 1.0/np.sqrt(5)
            elif n == 3:
                lattice_S[*[N_i//2 for N_i in N], :] = 1.0
        else:
            lattice_S[*[N_i//2 for N_i in N], :] = 1.0
    elif source_location.lower() == "corner":
        # Corner source:
        if n == 2:
            if m == 4:
                lattice_S[N[0]-1, 0, (0,3)] = 1.0
            elif m == 8:
                lattice_S[N[0]-1, 0, (0,3,7)] = 1.0
            elif m == 16:
                lattice_S[N[0]-1, 0, (0,3,7,14,15)] = 1.0
        elif n == 3:
            # @TODO - one (or maybe both?) of these is/are wrong
            if m == 16:
                lattice_S[N[0]-1, 0, 0, (0,3,7,13)] = 1.0
            elif m == 18:
                lattice_S[N[0]-1, 0, 0, (0,3,7,13)] = 1.0

    S_i = lattice_to_vector(lattice_S)

    alpha = 0.50/(m/4)
    angular_redistribution_coefficients = np.zeros((m,))
    angular_redistribution_coefficients[0] = alpha
    angular_redistribution_coefficients[+1] = (1-alpha)/2
    angular_redistribution_coefficients[-1] = (1-alpha)/2

    if n == 2:
        boundary_conditions = [("absorb", None)]*4
    elif n == 3:
        boundary_conditions = [("absorb", None)]*6

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        angular_redistribution_coefficients=angular_redistribution_coefficients,
        boundary_conditions=boundary_conditions,
        save_lattices=True, save_name="iso_src"
    )

    N_str = "-".join([str(N_i) for N_i in N])
    np.save(f"outputs/lattice_iso_src_{n_timesteps}_{N_str}_{m}.npy", lattices_I)

    analysis_isotropic_source(n=n, m=m, N=N, n_timesteps=n_timesteps)

def analysis_isotropic_source(n=2, m=8, N=None, n_timesteps=3, share_colorbar=None, filename=None, save_pdf=True, show=False):
    n = int(n)

    if N is None:
        N = [8, 8]
    elif isinstance(N, str):
        N = [int(d) for d in N.split(",")]

    N_str = "-".join([str(N_i) for N_i in N])

    m = int(m)

    n_timesteps = int(n_timesteps)

    share_colorbar = isinstance(share_colorbar, str) and share_colorbar.lower() in ["true", "1"]

    if filename is None or (isinstance(filename, str) and filename == ""):
        filename = f"outputs/lattice_iso_src_{n_timesteps}_{N_str}_{m}.npy"

    save_pdf = isinstance(save_pdf, str) and save_pdf.lower() in ["true", "1"]

    lattices = np.load(filename)

    if n_timesteps == 0:
        plot_timesteps = [0, 0]
    else:
        # plot_timesteps = sorted(set([*range(0, n_timesteps+1, int(np.ceil(n_timesteps/6))), n_timesteps]))
        plot_timesteps = list(range(n_timesteps+1))

    plot_lattices = [lattices[plot_timestep] for plot_timestep in plot_timesteps]

    print(plot_timesteps)

    ncols = len(plot_timesteps)

    # Plot 1: Full grid
    fig, ax = plt.subplots(ncols=ncols)
    
    if share_colorbar:
        max_val = np.max(lattices)

        for t, (timestep, lattice) in enumerate(zip(plot_timesteps, plot_lattices)):
            if n == 2:
                # Plot the entire lattice
                im = ax[timestep].imshow(lattice.sum(axis=2), vmax=max_val)
            elif n == 3:
                # @TODO - should be exposed as a user option
                # Plot the z=0 slice
                # im = ax[timestep].imshow(lattice.sum(axis=3)[:, 0, :], vmax=max_val)
                # Plot a projection onto the z=0 slice
                im = ax[timestep].imshow(lattice.sum(axis=3).sum(axis=2), vmax=max_val)

            ax[timestep].set_title(f"Timestep {timestep}")

        fig.subplots_adjust(bottom=0.1)
        cbar_ax = fig.add_axes([0.1, 0.15, 0.85, 0.05])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    else:
        for t, (timestep, lattice) in enumerate(zip(plot_timesteps, plot_lattices)):
            if n == 2:
                # Plot the entire lattice
                im = ax[t].imshow(lattice.sum(axis=2), norm=LogNorm(), cmap="magma")
            elif n == 3:
                # @TODO - should be exposed as a user option
                # Plot the z=0 slice
                # im = ax[timestep].imshow(lattice.sum(axis=3)[0, :, :])
                # Plot a projection onto the z=0 slice
                im = ax[t].imshow(lattice.sum(axis=3).sum(axis=2))
                
            fig.colorbar(im, orientation="horizontal", pad=0.1)
          
            ax[t].set_title(f"Timestep {timestep}")

    # Save PNG
    plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}.png"
    plt.savefig(plot_filename, dpi=256)
    print(f"Saved plot to {plot_filename}")

    # Save PDF
    if save_pdf:
        plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}.pdf"
        plt.savefig(plot_filename, format="pdf", dpi=256)
        print(f"Saved PDF to {plot_filename}")

    if show:
        plt.show()

    # Plot 2: Radial intensity profile scatterplot

    Rsq_list = []
    R_list = []
    I_list = []

    # @TODO - hardcoded center coordinates should instead take source location
    hx = N[0]//2
    hy = N[1]//2
    hz = 0 if n == 2 else N[2]//2

    NR = np.linalg.norm(N)

    if n == 2:
        for i in range(N[0]):
            for j in range(N[1]):
                Rsq = (i-hx)**2 + (j-hy)**2
                Rsq_list.append(Rsq)
                I_list.append(np.sum(lattices[-1][i, j]))
    elif n == 3:
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    Rsq = (i-hx)**2 + (j-hy)**2 + (k-hz)**2
                    Rsq_list.append(Rsq)
                    I_list.append(np.sum(lattices[-1][i, j, k]))

    Rsq_list = np.array(Rsq_list)
    R_list = np.sqrt(Rsq_list)
    I_list = np.array(I_list)

    # @TODO - normalize radii and/or intensities - can sometimes help (or hurt) fitting
    Rsq_list = Rsq_list / np.max(Rsq_list)
    R_list = R_list / np.max(R_list)
    I_list = I_list / np.max(I_list)

    fig, ax = plt.subplots()

    plt.scatter(R_list, I_list)

    plt.title("Radial intensity profile scatterplot")

    # Save PNG
    plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_intensity_scatter.png"
    plt.savefig(plot_filename, dpi=256)
    print(f"Saved plot to {plot_filename}")

    # Save PDF
    if save_pdf:
        plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_intensity_scatter.png"
        plt.savefig(plot_filename, format="pdf", dpi=256)
        print(f"Saved PDF to {plot_filename}")

    # Plot 3: Mean radial intensity profile

    # Collect intensities at the same radii
    I_dict = {}
    for Rsq, I in zip(Rsq_list, I_list):
        I_dict[Rsq] = I_dict.get(Rsq, []) + [I]

    Rsq_profile_list = []
    I_mean_list = []
    I_std_list = []
    for Rsq, I in zip(Rsq_list, I_list):
        Rsq_profile_list.append(Rsq)
        I_mean_list.append(np.mean(I_dict[Rsq]))
        I_std_list.append(np.std(I_dict[Rsq]))

    R_profile_list = np.sqrt(Rsq_profile_list)
    I_mean_list = np.array(I_mean_list)
    I_std_list = np.array(I_std_list)

    fig, ax = plt.subplots()

    plt.scatter(R_profile_list, I_mean_list)

    plt.title("Mean radial intensity profile scatterplot")

    # Save PNG
    plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_mean_intensity_scatter.png"
    plt.savefig(plot_filename, dpi=256)
    print(f"Saved PDF to {plot_filename}")

    # Save PDF
    if save_pdf:
        plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_mean_intensity_scatter.pdf"
        plt.savefig(plot_filename, format="pdf", dpi=256)
        print(f"Saved PDF to {plot_filename}")

    # Plot 4: Radial anisotropy (coefficient of variation) profile
    I_COV_list = I_std_list#/I_mean_list

    fig, ax = plt.subplots()

    plt.scatter(R_profile_list, I_COV_list)

    plt.title("Radial intensity anisotropy (COV) profile scatterplot")

    # Save PNG
    plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_intensity_COV_scatter.png"
    plt.savefig(plot_filename, dpi=400)
    print(f"Saved plot to {plot_filename}")

    # Save PDF
    if save_pdf:
        plot_filename = f"outputs/plot_iso_src_{n_timesteps}_{N_str}_{m}_intensity_COV_scatter.pdf"
        plt.savefig(plot_filename, format="pdf", dpi=400)
        print(f"Saved PDF to {plot_filename}")

# Shadow test
def test_shadow(n=2, m=8, source_type=None, N=None, n_timesteps=5, renormalize=False, lattice_I=None, lattice_S=None):
    n = int(n)
    if n != 2:
        raise ValueError(f"Shadow test isn't implemented for n={n}")

    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    M = np.prod(N)

    if lattice_I is None:
        I_i = np.zeros(shape=(M, m))
    else:
        if isinstance(lattice_I, str):
            lattice_I = np.load(lattice_I)
            lattice_I = lattice_I[-1]
        elif hasattr(lattice_I, "__iter__"):
            if np.shape(lattice_I) == (*N, m):
                lattice_I = lattice_I
            elif np.shape(lattice_I)[1:] == (*N, m):
                lattice_I = lattice_I[-1]
            else:
                raise ValueError(f"Could not resolve warm start lattice with shape {np.shape(lattice_I)}; expected (*N, m) or (*, *N, m)")
        else:
            raise TypeError()

        I_i = lattice_to_vector(lattice_I)

    if lattice_S is None:
        lattice_S = np.zeros(shape=(*N, m))
    else:
        if hasattr(lattice_S, "__iter__"):
            if np.shape(lattice_S) == (*N, m):
                lattice_S = lattice_S
            elif np.shape(lattice_S)[1:] == (*N, m):
                lattice_S = lattice_S[-1]
        elif isinstance(lattice_S, str):
            lattice_S = np.load(lattice_S)

    if source_type is None:
        source_type = "wall"
    else:
        source_type = source_type.lower()

    if "half" in source_type:
        N[1] *= 2

    if source_type == "point":
        # Option 1: Isotropic source near left wall
        if renormalize:
            if m == 4:
                lattice_S[2, N[1]//2, 0:4] = 1.0
            elif m == 8:
                lattice_S[2, N[1]//2, 0:4] = 1.0
                lattice_S[2, N[1]//2, 5:8] = 1.0/np.sqrt(2)
        else:
            lattice_S[2, N[1]//2, :] = 1.0

        R = 1
        a = 1/8 * N[0] # Half-width along x-axis
        b = 1/6 * N[1] # Half-width along y-axis
        h = -N[0] / 16
        k = 0
        
        save_name = "shadow-point"
    elif source_type == "half-point":
        # Option 2: Isotropic source near left wall, half-domain

        if m == 4:
            lattice_S[0, 0, (2,3)] = 1
        elif m == 8:
            lattice_S[0, 0, (2,3,6)] = 1
        elif m == 16:
            lattice_S[0, 0, (2,3,6,12,13)] = 1

        # lattice_S[1*N[0]//4, N[1]//4, :] = 1
        # lattice_S[1*N[0]//4, N[1]//2-1, :] = 1 # optional second source (for symmetry)

        R = 1
        a = 1/6 * N[0] # Half-width along x-axis
        b = 1/6 * N[1] # Half-width along y-axis
        h = -N[0] / 8
        k = 0
        
        save_name = "shadow-half-point"
    elif source_type == "wall":
        # Option 3: Construct source on left wall
        if m == 4:
            # Horizontal source:
            lattice_S[0, :, 2] = 1
        elif m == 8:
            # Horizontal source:
            lattice_S[0, :, 2] = 1

            # Diagonal sources:
            # lattice_S[0, :, 6] = 1
            # lattice_S[0, :, 7] = 1
        elif m == 16:
            # Horizontal source:
            lattice_S[0, :, 2] = 1

            # Diagonal sources:
            # lattice_S[0, :, 11] = 1
            # lattice_S[0, :, 12] = 1
        else:
            raise NotImplementedError(f"Shadow test isn't implemented for m={m}")

        R = 1
        a = 1/8 * N[0] # Half-width along x-axis b = 1/4 * N[1] # Half-width along y-axis
        h = -N[0] / 4
        k = 0
        
        save_name = "shadow-wall"
    elif source_type == "half-wall":
        # Option 4: Construct source on left wall, half-domain
        if m == 4:
            # Horizontal source:
            lattice_S[0, :, 2] = 1
        elif m == 8:
            # Horizontal source:
            lattice_S[0, :, 2] = 1

            # Diagonal sources:
            # lattice_S[0, :, 6] = 1
            # lattice_S[0, :, 7] = 1
        elif m == 16:
            # Horizontal source:
            lattice_S[0, :, 2] = 1

            # Diagonal sources:
            # lattice_S[0, :, 11] = 1
            # lattice_S[0, :, 12] = 1
        else:
            raise NotImplementedError(f"Shadow test isn't implemented for m={m}")

        R = 1
        a = 1/6 * N[0] # Half-width along x-axis
        b = 1/6 * N[1] # Half-width along y-axis
        h = -N[0] / 4
        k = 0
        
        save_name = "shadow-half-wall"
    else:
        raise ValueError(f"Input '{source_type}' is not a valid source type for the shadow test")

    print("Ellipse center:", h, k)
    print("Ellipse axes:", a, b)

    print(lattice_S)

    S_i = lattice_to_vector(lattice_S)

    # Construct opaque elliptical mask
    X, Y = np.meshgrid(range(N[0]), range(N[1]))
    opaque_ellipse = ((X - N[0]//2)/a) ** 2 + ((Y - N[1]//2)/b) ** 2 <= R**2
    opaque_ellipse = np.roll(opaque_ellipse, shift=[int(h - N[0]), int(k - N[1])], axis=[1,0])
    opaque_ellipse = opaque_ellipse.T

    # Use mask to construct opacity lattice
    kappa = np.zeros((N[0], N[1]))
    kappa[opaque_ellipse] = 1.0

    if "half" in source_type:
        kappa = kappa[:, N[1]//2:]

    print(kappa)

    if "half" in source_type:
        N[1] //= 2

    alpha = 0.33/(m/4)
    angular_redistribution_coefficients = np.zeros((m,))
    angular_redistribution_coefficients[0] = alpha
    angular_redistribution_coefficients[+1] = (1-alpha)/2
    angular_redistribution_coefficients[-1] = (1-alpha)/2

    boundary_conditions = [("absorb", None)]*4

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        kappa=kappa,
        angular_redistribution_coefficients=angular_redistribution_coefficients,
        boundary_conditions=boundary_conditions,
        save_lattices=True, save_name=save_name
    )

    np.save(f"outputs/lattice_shadow-{source_type}_{n_timesteps}_{N[0]}-{N[1]}.npy", lattices_I)

    analysis_shadow(source_type=source_type, N=N, n_timesteps=n_timesteps, renormalize=renormalize)

def analysis_shadow(n=UNUSED, m=UNUSED, source_type=None, N=None, n_timesteps=5, renormalize=UNUSED, filename=None, show=False):
    if source_type is None or source_type not in ["point", "half-point", "wall", "half-wall"]:
        raise ValueError(f"Input '{source_type}' is not a valid source type for the shadow test")

    if N is None:
        N = [8, 8]
    elif isinstance(N, str):
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    if filename is None:
        filename = f"outputs/lattice_shadow-{source_type}_{n_timesteps}_{N[0]}-{N[1]}.npy"

    lattices = np.load(filename)

    # Plot 1: Full grid
    if n_timesteps == 0 or len(lattices) == 1:
        plot_timesteps = [0, 0]
    else:
        # @TODO - should be exposed as a user option
        plot_timesteps = sorted(set([*range(0, n_timesteps+1, int(np.ceil(n_timesteps/8))), n_timesteps]))
        # plot_timesteps = range(n_timesteps+1)

    ncols = len(plot_timesteps)

    fig, ax = plt.subplots(ncols=ncols)

    # Track axes index t separately from timestep (since we only plot a subset of the timesteps)
    t = 0
    for timestep in plot_timesteps:
        lattice = lattices[timestep]

        ax[t].imshow(lattice.sum(axis=2), cmap="magma", norm=LogNorm())

        ax[t].set_title(f"Timestep {timestep}")

        t += 1

    plot_filename = f"outputs/plot_shadow-{source_type}_{N[0]}-{N[1]}_{n_timesteps}.png"
    plt.savefig(plot_filename, dpi=500)
    print(f"Saved plot to {plot_filename}")

    if show:
        plt.show()

    # Plot 2: Final radiation profile at far right boundary
    fig, ax = plt.subplots()

    yprofile = lattice.sum(axis=2)[-1, :]
    y = np.arange(len(yprofile))
    ax.scatter(y, yprofile)

    # @TODO - should be mutually exclusive (and exposed as a user option)
    # ax.set_ylim(bottom=0)
    ax.set_yscale("log")

    plt.title(f"Final radiation profile at far right boundary")

    plot_filename = f"outputs/plot_shadow-{source_type}_{N[0]}-{N[1]}_{n_timesteps}_final_radiation_profile.png"
    plt.savefig(plot_filename, dpi=500)
    print(f"Saved plot to {plot_filename}")

    if show:
        plt.show()

# Crossing radiation beams test
def test_crossing_radiation_beams(n=2, m=8, N=None, hw=None, beam_profile=None, n_timesteps=5, lattice_I=None, lattice_S=None):
    n = int(n)
    if n != 2:
        raise ValueError(f"Crossing radiation beams test isn't implemented for n={n}")

    m = int(m)

    if N is None:
        N = [8, 8]
    elif isinstance(N, str):
        N = [int(d) for d in N.split(",")]

    # Beam half-width computation
    if hw is None:
        hw = int(1/2 * N[0]//3)
    else:
        hw = int(hw)

    n_timesteps = int(n_timesteps)

    M = np.prod(N)

    if lattice_I is None:
        I_i = np.zeros(shape=(M, m))
    else:
        if isinstance(lattice_I, str):
            lattice_I = np.load(lattice_I)
            lattice_I = lattice_I[-1]
        elif hasattr(lattice_I, "__iter__"):
            if np.shape(lattice_I) == (*N, m):
                lattice_I = lattice_I
            elif np.shape(lattice_I)[1:] == (*N, m):
                lattice_I = lattice_I[-1]
            else:
                raise ValueError(f"Could not resolve warm start lattice with shape {np.shape(lattice_I)}; expected (*N, m) or (*, *N, m)")
        else:
            raise TypeError()

        I_i = lattice_to_vector(lattice_I)

    lattice_S = np.zeros(shape=(*N, m))

    if beam_profile == None or beam_profile == "uniform":
        beam_profile_callable = lambda jval, kval : 1
    elif beam_profile == "linear":
        beam_profile_callable = lambda jval, kval : jval - np.abs(kval)
    elif beam_profile == "gaussian":
        beam_profile_callable = lambda jval, kval : np.exp(-(kval/jval)**2)
    else:
        raise ValueError(f"Unrecognized input for beam_profile: '{beam_profile}'; please choose from None, 'uniform', 'linear', and 'gaussian'")

    for x in range(-hw, hw+1):
        if m == 8:
            lattice_S[0, N[1]//4+x, 7] = beam_profile_callable(hw, x)
            lattice_S[0, N[1]-1-N[1]//4+x, 4] = beam_profile_callable(hw, x)
        else:
            raise ValueError(f"Crossing radiation beams test isn't implemented for m={m}")

    S_i = lattice_to_vector(lattice_S)

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        save_lattices=True, save_name=f"crb-{beam_profile}"
    )

    lattices_filename = f"outputs/lattice_crb-{beam_profile}_{n_timesteps}_{N[0]}-{N[1]}_{hw}.npy"
    np.save(lattices_filename, lattices_I)
    print(f"Saved full data to {lattices_filename}")

    analysis_crossing_radiation_beams(N=N, hw=hw, beam_profile=beam_profile, n_timesteps=n_timesteps)

def analysis_crossing_radiation_beams(n=UNUSED, m=UNUSED, N=None, hw=None, beam_profile=None, n_timesteps=5, filename=None, save_pdf=True, show=False):
    if N is None:
        N = [8, 8]
    elif isinstance(N, str):
        N = [int(d) for d in N.split(",")]

    if hw is None:
        hw = int(1/2 * N[0]//3)
    elif isinstance(hw, str):
        hw = int(hw)

    n_timesteps = int(n_timesteps)

    if filename is None or (isinstance(filename, str) and filename == ""):
        filename = f"outputs/lattice_crb-{beam_profile}_{n_timesteps}_{N[0]}-{N[1]}_{hw}.npy"
    lattices = np.load(filename)

    save_pdf = isinstance(save_pdf, str) and save_pdf.lower() in ["true", "1"]

    if n_timesteps == 0:
        plot_timesteps = [0, 0]
    else:
        plot_timesteps = sorted(set([*range(0, n_timesteps+1, int(np.ceil(n_timesteps/6))), n_timesteps]))

    ncols = len(plot_timesteps)

    fig, ax = plt.subplots(ncols=ncols)

    # Track axes index t separately from timestep (since we only plot a subset of the timesteps)
    t = 0
    for timestep in plot_timesteps:
        lattice = lattices[timestep]

        ax[t].imshow(lattice.sum(axis=2))

        ax[t].set_title(f"Timestep {timestep}")

        t += 1

    # Save PNG
    plot_filename = f"outputs/plot_crb_{N[0]}-{N[1]}_{hw}_{n_timesteps}.png"
    plt.savefig(plot_filename, dpi=256)
    print(f"Saved plot to {plot_filename}")

    # Save PDF
    if save_pdf:
        plot_filename = f"outputs/plot_crb_{N[0]}-{N[1]}_{hw}_{n_timesteps}.pdf"
        plt.savefig(plot_filename, format="pdf", dpi=256)
        print(f"Saved PDF to {plot_filename}")

    if show:
        plt.show()

# Amplitude loss test
def test_amplitude_loss(n=1, m=2, N=None, n_timesteps=1000):
    n = int(n)
    m = int(m)

    if N is None:
        N = [4]
    else:
        N = [int(d) for d in N.split(",")]

    n_timesteps = int(n_timesteps)

    M = np.prod(N)

    I_i = np.zeros(shape=(M, m))
    I_i[[0]*n, 0] = 1 # in the sourceless free intensity case, does the amplitude actually tighten and/or increase over time?

    S_i = np.zeros(shape=(M, m))
    # S_i[[0]*n, 1] = 1

    lattices_I, lattices_S, norms = simulate(
        I_i, S_i,
        n, m,
        N,
        n_timesteps=n_timesteps,
        save_lattices=False, save_name="amp_loss"
    )

    timesteps = np.arange(n_timesteps+1)
    norms = [np.linalg.norm(lattice_I) for lattice_I in lattices_I]
    np.savetxt(f"outputs/norms_amp_loss_{n_timesteps}.txt", norms)

    analysis_amplitude_loss(n_timesteps=n_timesteps)

def analysis_amplitude_loss(n=UNUSED, m=UNUSED, N=UNUSED, n_timesteps=5, show=False):
    n_timesteps = int(n_timesteps)

    timesteps = np.arange(n_timesteps+1)
    norms = np.loadtxt(f"outputs/norms_amp_loss_{n_timesteps}.txt")

    print(f"Mean: {np.mean(norms[1:])}\nStandard Deviation: {np.std(norms[1:])}\nRange: {np.ptp(norms[1:])}")

    a, b = np.polyfit(timesteps[1:], norms[1:], 1)
    print(f"Line of best fit: A(t) ≈ {a:.6f}t + {b:.6f}")

    fig, ax = plt.subplots()
    plt.scatter(timesteps, norms)
    plt.plot(timesteps, a * timesteps + b)

    if show:
        plt.show()

if __name__ == "__main__":
    func = sys.argv[1]
    args = sys.argv[2:]
    print(f"running {func}({args})...")

    # Process any mix of positional and keyword args
    pargs = []
    kwargs = {}

    for arg in args:
        if "=" in arg:
            arg_key = arg.split("=")[0]
            arg_val = arg.split("=")[1]
            kwargs[arg_key] = arg_val
        else:
            pargs.append(arg)

    locals()[func](*pargs, **kwargs)

