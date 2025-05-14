import sys

import numpy as np
import matplotlib.pyplot as plt

from qlbm_rt import simulate
from qlbm_utils import lattice_to_vector

# Point source 1D test
def test_point_source_1D(N=None, n_it=10):
    n = 1
    m = 2

    if N is None:
        N = [8]
    else:
        N = [int(N)]

    n_it = int(n_it)

    M_0 = np.prod(N)
    M = 2**int(np.ceil(np.log2(M_0)))

    I_i = np.zeros(shape=(M, m))
    I_i[N[0]//2, :] = 1

    src_lattice = np.zeros(shape=(*N, m))
    # src_lattice[N[0]//2, :] = 1
    S_i = lattice_to_vector(src_lattice)

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        save_lattices=True
    )

    lattices = np.save(f"outputs/lattice_point_src_1D_{n_it}.npy", lattices)

    analysis_point_source_1D(n_it)

def analysis_point_source_1D(n_it=10, sep_dirs=True):
    n_it = int(n_it)

    lattices = np.load(f"outputs/lattice_point_src_1D_{n_it}.npy")

    fig, ax = plt.subplots(nrows=n_it+1)

    for it, lattice in enumerate(lattices):
        if sep_dirs:
            ax[it].scatter(np.arange(lattice.shape[0]), lattice[:, 0], c="red")
            ax[it].scatter(np.arange(lattice.shape[0]), lattice[:, 1], c="blue")
        else:
            ax[it].scatter(np.arange(lattice.shape[0]), np.sum(lattice, axis=1))

        ax[it].set_title(f"Iteration {it}")

    plt.show()

# Unidirectional radiation test
def test_unidirectional_source(n=2, m=8, N=None, n_it=3):
    n = int(n)
    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_it = int(n_it)

    M = np.prod(N)

    # @TODO - apply multiprocessing
    lattices = [] # shape (mu, n_it+1, *N, m)
    for mu in range(m):
        I_i = np.zeros(shape=(M, m))

        src_lattice = np.zeros(shape=(*N, m))
        src_lattice[*[Ni//2 for Ni in N], mu] = 1
        S_i = lattice_to_vector(src_lattice)

        lattices_mu = simulate(
            I_i, S_i,
            n, m,
            N,
            n_it=n_it,
            save_lattices=True
        )

        lattices.append(lattices_mu)

    np.save(f"outputs/lattice_uni_src_{n_it}_{m}.npy", lattices)

    analysis_unidirectional_source(m, n_it)

def analysis_unidirectional_source(m=8, n_it=3):
    m = int(m)
    n_it = int(n_it)

    fig, ax = plt.subplots(figsize=(6,18), nrows=m, ncols=n_it+1)

    lattices = np.load(f"outputs/lattice_uni_src_{n_it}_{m}.npy", allow_pickle=True)
    N = lattices.shape[2:-1]
    print(lattices.shape)
    print(lattices.shape[2:-1])
    for mu in range(m):
        print("/"*20 + " " + str(mu) + " " + "/"*20)
        lattices_mu = lattices[mu]
        for it, lattice in enumerate(lattices_mu):
            spwn = lattice[:,:,mu]
            sumspwn = np.sum(spwn)
            sums = np.sum(lattice)
            lttc = np.sum(np.sum(lattice, axis=0), axis=0)
            print(it, sumspwn, sums, "-", lttc)

            lattice[*[Ni//2 for Ni in N] ,:] = 0.25 * np.sum(lattice)
            ax[mu][it].imshow(lattice.sum(axis=2).T)

            ax[mu][it].set_title(f"Direction {mu} - Iteration {it}")

    plt.savefig(f"outputs/plot_uni_src_{n_it}_{m}.png", dpi=500)
    plt.show()

# Angular redistribution spread test
def test_redistribution_spread(n=2, m=8, mu_0=0, alpha=0.33, N=None, n_it=3):
    n = int(n)
    m = int(m)

    mu_0 = int(mu_0)
    alpha = float(alpha)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_it = int(n_it)

    M = np.prod(N)
    angular_redistribution_coefficients = np.zeros((m,m))

    for mu in range(m):
        mu_next = (mu + 1) % m
        mu_prev = (mu - 1) % m

        angular_redistribution_coefficients[mu, mu] = alpha
        angular_redistribution_coefficients[mu, mu_next] = (1-alpha)/2
        angular_redistribution_coefficients[mu, mu_prev] = (1-alpha)/2

    I_i = np.zeros(shape=(M, m))

    src_lattice = np.zeros(shape=(*N, m))
    src_lattice[*[Ni//2 for Ni in N], mu_0] = 1
    S_i = lattice_to_vector(src_lattice)

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        angular_redistribution_coefficients=angular_redistribution_coefficients,
        save_lattices=True
    )

    np.save(f"outputs/lattice_redist_spread_{n_it}_{m}_{mu_0}.npy", lattices)

    analysis_redistribution_spread(m, mu_0, n_it)

def analysis_redistribution_spread(m=8, mu_0=0, n_it=3):
    n_it = int(n_it)

    lattices = np.load(f"outputs/lattice_redist_spread_{n_it}_{m}_{mu_0}.npy")

    fig, ax = plt.subplots(ncols=n_it+1)

    for it, lattice in enumerate(lattices):
        ax[it].imshow(lattice.sum(axis=2).T[:, ::-1])

        ax[it].set_title(f"Iteration {it}")

    plt.show()

# Isotropic source test
def test_isotropic_source(n=2, m=8, N=None, n_it=3):
    n = int(n)
    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_it = int(n_it)

    M = np.prod(N)

    I_i = np.zeros(shape=(M, m))

    src_lattice = np.zeros(shape=(*N, m))
    src_lattice[*[N_i//2 for N_i in N], :] = 1
    plt.imshow(np.sum(src_lattice, axis=2).T)

    S_i = lattice_to_vector(src_lattice)

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        save_lattices=True
    )

    np.save(f"outputs/lattice_iso_source_{n_it}.npy", lattices)

    analysis_isotropic_source(n_it)

def analysis_isotropic_source(n_it=3):
    n_it = int(n_it)

    lattices = np.load(f"outputs/lattice_iso_source_{n_it}.npy")

    fig, ax = plt.subplots(ncols=n_it+1)

    for it, lattice in enumerate(lattices):
        ax[it].imshow(lattice.sum(axis=2).T)

        ax[it].set_title(f"Iteration {it}")

    plt.show()

# Crossing radiation beams test
def test_crossing_radiation_beams(n=2, m=8, N=None, hw=None, n_it=5):
    n = int(n)
    if n != 2:
        raise ValueError(f"Crossing radiation beams test isn't implemented for n={n}")

    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    # Beam half-width computation
    if hw is None:
        hw = int(1/2 * N[0]//3)
    else:
        hw = int(hw)

    n_it = int(n_it)

    M = np.prod(N)

    I_i = np.zeros(shape=(M, m))

    src_lattice = np.zeros(shape=(*N, m))

    beam_profile_linear = lambda jval, kval : jval - np.abs(kval)
    beam_profile_normal = lambda jval, kval : np.exp(-k**2)

    for x in range(-hw, hw+1):
        if m == 8:
            src_lattice[1*N[0]//3+x, N[1]-2, 5] = beam_profile_linear(hw, x)
            src_lattice[2*N[0]//3+x, N[1]-2, 4] = beam_profile_linear(hw, x)
        elif m == 16:
            src_lattice[1*N[0]//4+x, N[1]-1, 6] = hw-np.abs(x)
            src_lattice[3*N[0]//4+x, N[1]-1, 7] = hw-np.abs(x)

    plt.imshow(np.sum(src_lattice, axis=2).T)

    S_i = lattice_to_vector(src_lattice)

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        save_lattices=True
    )

    np.save(f"outputs/lattice_crb_{n_it}_{N[0]}-{N[1]}_{hw}.npy", lattices)

    analysis_crossing_radiation_beams(n_it, N, hw)

def analysis_crossing_radiation_beams(N=None, hw=None, n_it=5):
    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    if hw is None:
        hw = int(1/2 * N[0]//3)
    else:
        hw = int(hw)

    n_it = int(n_it)

    lattices = np.load(f"outputs/lattice_crb_{n_it}_{N[0]}-{N[1]}_{hw}.npy")

    ncols = min(n_it+1, 6)
    fig, ax = plt.subplots(ncols=ncols)

    for it in range(n_it+1):
        print(it)
        if n_it <= 5 or (it % ncols) == 0:
            lattice = lattices[it]

            ax[it].imshow(lattice.sum(axis=2).T)

            ax[it].set_title(f"Iteration {it}")

    plt.show()

# Shadow test
def test_shadow(n=2, m=8, source_type=None, N=None, n_it=5):
    n = int(n)
    if n != 2:
        raise ValueError(f"Shadow test isn't implemented for n={n}")

    m = int(m)

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_it = int(n_it)

    M = np.prod(N)

    I_i = np.zeros(shape=(M, m))

    src_lattice = np.zeros(shape=(*N, m))

    if source_type is not None and source_type.lower() == "point":
        # Option 1: Isotropic source near left wall
        src_lattice[1*N[0]//4, N[1]//2] = 2

        R = np.sqrt((N[0]**2+N[1]**2)/4)
        a = 0.05
        b = 0.1
        h = N[0] / 3
        k = N[1] / 2
    else:
        # Option 2 (default): Construct source on left wall
        source_type = "wall"
        if m == 8:
            # src_lattice[0, :, 0] = 1
            src_lattice[0, :, 4] = 1
            src_lattice[0, :, 7] = 1
        elif m == 16:
            # src_lattice[0, :, 0] = 1
            src_lattice[0, :, 7] = 1
            src_lattice[0, :, 15] = 1
        else:
            raise ValueError(f"Shadow test isn't implemented for m={m}")

        R = np.sqrt((N[0]**2+N[1]**2)/2)
        a = 0.05
        b = 0.1
        h = N[0] / 2
        k = N[1] / 2

    S_i = lattice_to_vector(src_lattice)
    
    # Construct opaque ellipse
    X, Y = np.meshgrid(range(N[0]), range(N[1]))
    opaque_ellipse = ((X - h)/a) ** 2 + ((Y - k)/b) ** 2 <= R**2
    kappa = np.zeros((N[0], N[1]))
    kappa[opaque_ellipse] = 1
    plt.imshow(kappa)

    boundary_conditions = [("absorb", None)]*4

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        kappa=kappa,
        boundary_conditions=boundary_conditions,
        save_lattices=True
    )

    np.save(f"outputs/lattice_shadow-{source_type}_{n_it}_{N[0]}-{N[1]}.npy", lattices)

    analysis_shadow(source_type, N, n_it)

def analysis_shadow(source_type=None, N=None, n_it=5):
    if source_type is not None and source_type.lower() == "point":
        source_type = "point"
    else:
        source_type = "wall"

    if N is None:
        N = [8, 8]
    else:
        N = [int(d) for d in N.split(",")]

    n_it = int(n_it)

    lattices = np.load(f"outputs/lattice_shadow-{source_type}_{n_it}_{N[0]}-{N[1]}.npy")

    fig, ax = plt.subplots()

    ax.imshow(lattice[-1].sum(axis=2).T)
    ax.set_title(f"Radiation intensity - Iteration {it}")

    plt.show()

# Amplitude loss test
def test_amplitude_loss(n=1, m=2, N=None, n_it=1000):
    n = int(n)
    m = int(m)

    if N is None:
        N = [4]
    else:
        N = [int(d) for d in N.split(",")]

    M = np.prod(N)

    I_i = np.zeros(shape=(M, m))
    I_i[[0]*n, 0] = 1 # in the sourceless free intensity case, does the amplitude actually tighten and/or increase over time?

    S_i = np.zeros(shape=(M, m))
    # S_i[[0]*n, 1] = 1

    lattices = simulate(
        I_i, S_i,
        n, m,
        N,
        n_it=n_it,
        save_lattices=True
    )

    iterations = np.arange(n_it+1)
    norms = [np.linalg.norm(lattice) for lattice in lattices]
    np.savetxt(f"outputs/norms_amp_loss_{n_it}.txt", norms)

    analysis_amplitude_loss(n_it)

def analysis_amplitude_loss(n_it=5):
    n_it = int(n_it)

    iterations = np.arange(n_it+1)
    norms = np.loadtxt(f"outputs/norms_amp_loss_{n_it}.txt")

    print(f"Mean: {np.mean(norms[1:])}\nStandard Deviation: {np.std(norms[1:])}\nRange: {np.ptp(norms[1:])}")

    a, b = np.polyfit(iterations[1:], norms[1:], 1)
    print(f"Line of best fit: A(t) ≈ {a:.6f}t + {b:.6f}")

    fig, ax = plt.subplots()
    plt.scatter(iterations, norms)
    plt.plot(iterations, a * iterations + b)
    plt.show()

if __name__ == "__main__":
    option = sys.argv[1]
    args = sys.argv[2:]
    print(f"running {option}({args})...")

    locals()[option](*args)

