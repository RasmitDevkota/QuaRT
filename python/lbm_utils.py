import numpy as np

def compute_grid_parameters(n, N, verbose=False):
    M_0 = np.prod(N)

    # QuaRT LBM currently only supports lattices with resolutions which are powers of 2
    M = 2**int(np.ceil(np.log2(M_0)))

    if verbose:
        print(f"Number of lattice points: {M_0}")
        print(f"Next power of 2: {M}")

    return M_0, M

def compute_scheme_velocities(n, m):
    match n:
        case 1:
            if m == 2:
                idxs_dir = [0,  1]
                cxs      = [1, -1]
                cs       = cxs
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 1D lattices. Please choose from: m=2")
        case 2:
            if m == 4: # TESTING PURPOSES ONLY: In D2Q8, 0 2 5 7 work, 1 3 4 6 don't work as expected
                idxs_dir = [1,  3, 4,  6]
                cxs      = [0,  0, 1, -1]
                cys      = [1, -1, 1, -1]
            elif m == 8:
                idxs_dir = [0, 4,  1,  5, 2,  6,  3,  7]
                cxs      = [1, 0, -1,  0, 1, -1, -1,  1]
                cys      = [0, 1,  0, -1, 1,  1, -1, -1]
            elif m == 16:
                idxs_dir = [0, 4,  1,  5, 2,  6,  3,  7, 8, 9, 10, 11, 12, 13, 14, 15]
                cxs      = [1, 0, -1,  0, 1, -1, -1,  1, 2, 1, -1, -2, -2, -1,  1,  2]
                cys      = [0, 1,  0, -1, 1,  1, -1, -1, 1, 2,  2,  1, -1, -2, -2, -1]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 2D lattices. Please choose from: m=8, m=16")

            cs = list(zip(cxs, cys))
        case 3:
            if m == 18:
                idxs_dir = [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                cxs      = [-1,  0,  0, -1, -1, -1, -1,  0,  0, 1,  0,  0,  1,  1,  1,  1,  0,  0]
                cys      = [ 0, -1,  0, -1,  1,  0,  0, -1, -1, 0,  1,  0,  1, -1,  0,  0,  1,  1]
                czs      = [ 0,  0, -1,  0,  0, -1,  1, -1,  1, 0,  0,  1,  0,  0,  1, -1,  1, -1]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 3D lattices. Please choose from: m=18")

            cs = list(zip(cxs, cys, czs))

    return idxs_dir, cs

def compute_scheme_boundaries(n, m):
    match n:
        case 1:
            if m == 2:
                boundary_idxs_left = [0]
                boundary_idxs_right = [1]

                boundary_idxs = [boundary_idxs_left, boundary_idxs_right]
            else:
                raise ValueError("Only m=2 is supported for 1D lattices. Did you mean to pick a higher-dimensional lattice?")
        case 2:
            if m == 4:
                boundary_idxs_left = []
                boundary_idxs_right = []
                boundary_idxs_top = []
                boundary_idxs_bottom = []
            elif m == 8:
                boundary_idxs_left = [2, 5, 6]
                boundary_idxs_right = [0, 4, 7]
                boundary_idxs_top = [1, 4, 5]
                boundary_idxs_bottom = [3, 6, 7]
            elif m == 16:
                boundary_idxs_left = [2, 5, 6, 11, 12]
                boundary_idxs_right = [0, 4, 7, 8, 15]
                boundary_idxs_top = [1, 4, 5, 9, 10]
                boundary_idxs_bottom = [3, 6, 7, 13, 14]

            boundary_idxs = [boundary_idxs_left, boundary_idxs_right, boundary_idxs_top, boundary_idxs_bottom]
        case 3:
            # @TODO - come up with systematic/programmatic way of generating these
            boundary_idxs_ppp = boundary_idxs_mpp = boundary_idxs_mmp = boundary_idxs_pmp = boundary_idxs_ppm = boundary_idxs_mpm = boundary_idxs_mmm = boundary_idxs_pmm = []

            boundary_idxs = [
                boundary_idxs_ppp, boundary_idxs_mpp, boundary_idxs_mmp, boundary_idxs_pmp,
                boundary_idxs_ppm, boundary_idxs_mpm, boundary_idxs_mmm, boundary_idxs_pmm
            ]

    return boundary_idxs

