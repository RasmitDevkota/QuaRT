import numpy as np

def compute_grid_parameters(n, N, verbose=False):
    """Compute parameters of the simulation grid.
    """

    M_0 = np.prod(N)

    # QuaRT LBM currently only supports lattices with resolutions which are powers of 2
    M = 2**int(np.ceil(np.log2(M_0)))

    if verbose:
        print(f"Number of lattice points: {M_0}")
        print(f"Next power of 2: {M}")

    return M_0, M

def compute_scheme_velocities(n, m):
    """Compute velocities along angular directions of lattice scheme.
    """

    match n:
        case 1:
            if m == 2:
                idxs_dir = [0,  1]
                cxs      = [1, -1]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 1D lattices. Please choose from: m=2")

            cs = cxs
        case 2:
            if m == 4:
                idxs_dir = [0, 1,  2,  3]
                cxs      = [1, 0, -1,  0]
                cys      = [0, 1,  0, -1]
            elif m == 8:
                # @TODO - check which one is correct
                # idxs_dir = [0, 1,  2,  3, 4,  5,  6,  7]
                idxs_dir = [0, 4,  1,  5, 2,  6,  3,  7]
                cxs      = [1, 0, -1,  0, 1, -1, -1,  1]
                cys      = [0, 1,  0, -1, 1,  1, -1, -1]
            elif m == 16:
                idxs_dir = [0, 4,  1,  5, 2,  6,  3,  7, 8, 9, 10, 11, 12, 13, 14, 15]
                cxs      = [1, 0, -1,  0, 1, -1, -1,  1, 2, 1, -1, -2, -2, -1,  1,  2]
                cys      = [0, 1,  0, -1, 1,  1, -1, -1, 1, 2,  2,  1, -1, -2, -2, -1]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 2D lattices. Please choose from: m=4, m=8, m=16")

            cs = list(zip(cxs, cys))
        case 3:
            if m == 16:
                idxs_dir = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
                cxs      = [-1,  0,  1,  0, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1]
                cys      = [ 0, -1,  0,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1]
                czs      = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1, -1, -1, -1, -1]
            elif m == 18:
                idxs_dir = [ 0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                cxs      = [-1,  0,  0, -1, -1, -1, -1,  0,  0, 1,  0,  0,  1,  1,  1,  1,  0,  0]
                cys      = [ 0, -1,  0, -1,  1,  0,  0, -1, -1, 0,  1,  0,  1, -1,  0,  0,  1,  1]
                czs      = [ 0,  0, -1,  0,  0, -1,  1, -1,  1, 0,  0,  1,  0,  0,  1, -1,  1, -1]
            elif m == 32:
                idxs_dir = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                cxs      = [ 1,  0, -1,  0,  1, -1, -1,  1,  0,  1,  0, -1,  0,  1, -1, -1,  1,  0,  1,  0, -1,  0,  1, -1, -1,  1,  2,  0,  0, -2,  0,  0]
                cys      = [ 0,  1,  0, -1,  1,  1, -1, -1,  0,  0,  1,  0, -1,  1,  1, -1, -1,  0,  0,  1,  0, -1,  1,  1, -1, -1,  0,  2,  0,  0, -2,  0]
                czs      = [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  2,  0,  0, -2]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 3D lattices. Please choose from: m=16, m=18, m=32")

            cs = list(zip(cxs, cys, czs))
        case _:
            raise ValueError(f"Invalid dimension {n}")

    return idxs_dir, cs

def compute_scheme_adjacencies(n, m, idxs_dir, cs):
    """Construct matrix indicating angular direction adjacencies.
    """

    match n:
        case 1:
            if m == 2:
                adjacencies = [
                    [0,1],
                    [1,0]
                ]
            else:
                raise ValueError(f"Angular discretization m={m} is not supported for 1D lattices. Please choose from: m=2")
        case 2:
            adjacencies = np.zeros((m, m))
            for mu, c in enumerate(cs):
                c = np.array(c)
                for nu, d in enumerate(cs):
                    d = np.array(d)
                    if mu == nu:
                        adjacencies[mu][nu] = -1
                    else:
                        norm_inf = max(abs(c-d))
                        if norm_inf <= 1:
                            adjacencies[mu][nu] = +1
                        else:
                            adjacencies[mu][nu] = 0
        case 3:
            adjacencies = -1 * np.eye(m)

            for mu_0 in range(0, m-1):
                for mu_adj in range(mu_0+1, m):
                    if np.isclose(np.sum(np.abs(np.subtract(cs[mu_0], cs[mu_adj]))), 1.0):
                        adjacencies[mu_0, mu_adj] = +1
                        adjacencies[mu_adj, mu_0] = +1

            print(adjacencies)
        case _:
            raise ValueError(f"Invalid dimension {n}")

    return adjacencies

def compute_scheme_boundaries(n, m, idxs_dir=None, cs=None):
    """Construct lists of directions along each boundary of the simulation grid.

    IMPORTANT - Left and right boundary indices are switched because of numpy array indexing (technically these are top and bottom in the numpy array, but we transpose in post-processing).
    """

    match n:
        case 1:
            match m:
                case 2:
                    boundary_idxs_left = [1]
                    boundary_idxs_right = [0]

                    boundary_idxs = [boundary_idxs_left, boundary_idxs_right]
                case _:
                    raise ValueError("Only m=2 is supported for 1D lattices. Did you mean to pick a higher-dimensional lattice?")
        case 2:
            match m:
                case 4:
                    boundary_idxs_left = [0]
                    boundary_idxs_right = [2]
                    boundary_idxs_top = [1]
                    boundary_idxs_bottom = [3]
                case 8:
                    boundary_idxs_left = [0, 4, 7]
                    boundary_idxs_right = [2, 5, 6]
                    boundary_idxs_top = [1, 4, 5]
                    boundary_idxs_bottom = [3, 6, 7]
                case 16:
                    boundary_idxs_left = [0, 4, 7, 8, 15]
                    boundary_idxs_right = [2, 5, 6, 11, 12]
                    boundary_idxs_top = [1, 4, 5, 9, 10]
                    boundary_idxs_bottom = [3, 6, 7, 13, 14]
                case _:
                    raise ValueError(f"Angular discretization m={m} is not supported for 2D lattices. Please choose from: m=4, m=8, m=16")

            boundary_idxs = [boundary_idxs_left, boundary_idxs_right, boundary_idxs_top, boundary_idxs_bottom]
        case 3:
            boundary_idxs_xp = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[0] == +1]
            boundary_idxs_xm = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[0] == -1]
            boundary_idxs_yp = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[1] == +1]
            boundary_idxs_ym = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[1] == -1]
            boundary_idxs_zp = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[2] == +1]
            boundary_idxs_zm = [idx_dir for idx_dir, cs in zip(idxs_dir, cs) if cs[2] == -1]
                
            boundary_idxs = [boundary_idxs_xp, boundary_idxs_xm, boundary_idxs_yp, boundary_idxs_ym, boundary_idxs_zp, boundary_idxs_zm]

    return boundary_idxs

