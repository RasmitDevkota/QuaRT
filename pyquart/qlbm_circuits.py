import os
import time

import numpy as np
np.set_printoptions(threshold=np.inf)

import pickle

import multiprocessing as mp
from multiprocessing import Pool

from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import UnitaryGate, XGate, ZGate

def state_preparation(
    I_i, S_i,
    n, m,
    N,
    kappa,
    delta_t,
    coord_idx_map, m_max_bin,
    boundary_idxs, boundary_conditions,
    n_qubits, n_qubits_ancilla,
    qreg_lattice, qreg_mask, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    initial_statevector = np.zeros((2**n_qubits))

    anc_bin = "0"*n_qubits_ancilla

    prob_amp_I = 0
    prob_amp_S = 0

    if verbose:
        print("actually recovering intensities and sources:\n", I_i, "\n", S_i)

    S_i_coords = S_i.sum(axis=-1).reshape([Ni for Ni in N])

    source_coordinates = np.argwhere(S_i_coords > 0.0)
    
    if qreg_ancilla.size > 0 and qreg_mask.size > 0:
        radius_AR = np.mean(N[:2])//4 + 0.5
        print(radius_AR)

    for c, (coordinate, coordinate_bin) in enumerate(coord_idx_map.items()):
        coordinate_bin = coordinate_bin[::-1]

        if qreg_ancilla.size > 0 and qreg_mask.size > 0:
            mask_bool = np.any([(coordinate[0] - source_coordinate[0])**2 + (coordinate[1] - source_coordinate[1])**2 <= radius_AR**2 for source_coordinate in source_coordinates])
            mask_bin = str(int(mask_bool))

            if verbose:
                if mask_bool:
                    print(coordinate)
        else:
            mask_bin = ""

        for mu in range(m):
            mu_bin = bin(mu)[2:].zfill(len(m_max_bin))[::-1]

            for s_bin in range(2):
                boundary_bin = "0"

                # @TODO - handle source terms and simulations without special BCs separately
                # Check for wall boundary conditions
                match n:
                    case 1:
                        boundary_bin = "1" if (
                            (coordinate[0] == 0 and mu in boundary_idxs[0] and boundary_conditions[0][0] == "absorb") or
                            (coordinate[0] == N[0]-1 and mu in boundary_idxs[1] and boundary_conditions[1][0] == "absorb")
                        ) else "0"
                    case 2:
                        boundary_bin = "1" if (
                            (coordinate[0] == 0 and mu in boundary_idxs[0] and boundary_conditions[0][0] == "absorb") or
                            (coordinate[0] == N[0]-1 and mu in boundary_idxs[1] and boundary_conditions[1][0] == "absorb") or
                            (coordinate[1] == 0 and mu in boundary_idxs[2] and boundary_conditions[2][0] == "absorb") or
                            (coordinate[1] == N[1]-1 and mu in boundary_idxs[3] and boundary_conditions[3][0] == "absorb")
                        ) else "0"
                    case 3:
                        boundary_bin = "1" if (
                            (coordinate[0] == 0 and mu in boundary_idxs[0] and boundary_conditions[0][0] == "absorb") or
                            (coordinate[0] == N[0]-1 and mu in boundary_idxs[1] and boundary_conditions[1][0] == "absorb") or
                            (coordinate[1] == 0 and mu in boundary_idxs[2] and boundary_conditions[2][0] == "absorb") or
                            (coordinate[1] == N[1]-1 and mu in boundary_idxs[3] and boundary_conditions[3][0] == "absorb") or
                            (coordinate[2] == 0 and mu in boundary_idxs[4] and boundary_conditions[4][0] == "absorb") or
                            (coordinate[2] == N[2]-1 and mu in boundary_idxs[5] and boundary_conditions[5][0] == "absorb")
                        ) else "0"
                    case _:
                        raise ValueError(f"Invalid dimension {n}")

                # Treat fully-opaque interior coordinates as absorbing boundaries
                kappa_coord = kappa[coordinate] if hasattr(kappa, "__iter__") else kappa
                if (
                    np.isclose(kappa_coord, 1.0)
                ):
                    boundary_bin = "1"

                if s_bin == 0:
                    prob_amp = I_i[c, mu]
                    prob_amp_I += prob_amp
                else:
                    prob_amp = 1/m * delta_t * S_i[c, mu]
                    prob_amp_S += prob_amp

                idx_bin = f"0b{anc_bin}{s_bin}{mu_bin}{boundary_bin}{mask_bin}{coordinate_bin}"
                idx_dec = int(idx_bin, 2)
                initial_statevector[idx_dec] = prob_amp

    print("Total probability amplitude in intensities:", prob_amp_I)
    print("Total probability amplitude in sources:", prob_amp_S)

    norm = np.linalg.norm(initial_statevector)
    print("Initial statevector norm:", norm)

    if norm > 0:
        initial_statevector /= norm

    qc = QuantumCircuit(qreg_lattice, qreg_mask, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla)
    qc.initialize(initial_statevector)
    
    return qc, norm

def absorption_scattering(
    kappa, sigma,
    delta_t,
    I_2,
    n_qubits_direction,
    qreg_direction, qreg_switch, qreg_ancilla,
    ancilla_idxs_AS,
):
    # Hardcoded parameters for testing
    kappa = 0.0
    sigma = 0.0
    delta_t = 1

    # LCU method
    print("Intermediate quantities:")
    a0 = 1 - kappa * delta_t + 0.5 * sigma * delta_t
    a1 = 0.5 * sigma * delta_t
    print(a0, a1)
    print(a0-a1, a0+a1)
    print(1 - (a0 - a1)**2, 1 - (a0 + a1)**2)
    b0 = np.sqrt(1 - (a0 - a1)**2) + np.sqrt(1 - (a0 + a1)**2)
    b1 = np.sqrt(1 - (a0 - a1)**2) - np.sqrt(1 - (a0 + a1)**2)
    print(b0, b1)

    C_1 = np.array([
        [a0 + 0.5j * b0, a1 + 0.5j * b1],
        [a1 + 0.5j * b1, a0 + 0.5j * b0]
    ])

    C_2 = np.array([
        [a0 - 0.5j * b0, a1 - 0.5j * b1],
        [a1 - 0.5j * b1, a0 - 0.5j * b0]
    ])

    # Construct quantum operations
    C_1_gate = UnitaryGate(C_1, label="$C_1$").control(2)
    C_2_gate = UnitaryGate(C_2, label="$C_2$").control(2)

    # Perform quantum circuit composition steps
    qc = QuantumCircuit(qreg_direction, qreg_switch, qreg_ancilla)

    a = ancilla_idxs_AS[0]

    qc.h(qreg_ancilla[a])

    for q in range(n_qubits_direction):
        qc.x(qreg_switch[:])

        qc.x(qreg_ancilla[a])
        qc.append(C_1_gate, [qreg_ancilla[a]] + qreg_switch[:] + [qreg_direction[q]])
        qc.x(qreg_ancilla[a])

        qc.append(C_2_gate, [qreg_ancilla[a]] + qreg_switch[:] + [qreg_direction[q]])
        qc.x(qreg_switch[:])

    qc.h(qreg_ancilla[a])

    return qc

def absorption_emission(
    I_2M, Z_2M,
    qreg_switch, qreg_ancilla,
    ancilla_idxs_AE,
):
    # LCU method
    D_1 = XGate()
    D_2 = XGate()
    D_3 = ZGate()

    # Construct quantum operations
    D_1_gate = XGate().control(2)
    D_2_gate = XGate().control(2)
    D_3_gate = ZGate().control(2)

    # Perform quantum circuit composition steps
    qc = QuantumCircuit(qreg_switch, qreg_ancilla)

    qc.h(qreg_ancilla[ancilla_idxs_AE])

    qc.x(qreg_ancilla[ancilla_idxs_AE[1]])
    qc.append(D_1_gate, qreg_ancilla[ancilla_idxs_AE] + qreg_switch[:])
    qc.x(qreg_ancilla[ancilla_idxs_AE[1]])

    qc.append(D_2_gate, qreg_ancilla[ancilla_idxs_AE] + qreg_switch[:])

    qc.append(D_3_gate, qreg_ancilla[ancilla_idxs_AE] + qreg_switch[:])

    qc.h(qreg_ancilla[ancilla_idxs_AE])

    return qc

def angular_redistribution(
    m,
    delta_t,
    angular_redistribution_coefficients,
    idxs_dir, cs,
    adjacencies,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_mask, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_mask, qreg_direction, qreg_switch, qreg_ancilla,
    ancilla_idxs_AR
):
    # Construct matrices
    dim_R = 2**(n_qubits_direction)

    R_p = np.zeros((dim_R, dim_R))
    R_m = np.zeros((dim_R, dim_R))

    for mu_0 in range(m):
        # The adjacent indices can be accessed using integer deltas
        mu_0_idx = idxs_dir[mu_0]

        # Neat indexing trick so that adjacent directions are at indices +1 and -1
        idx_deltas = [0, +1, -1]

        mu_adj_idxs = [(mu_0_idx + idx_delta) % m for idx_delta in idx_deltas]
        mu_adjs = [idxs_dir[mu_adj_idx] for mu_adj_idx in mu_adj_idxs]
        mu_adj_bins = [bin(mu_adj)[2:].zfill(len(m_max_bin)) for mu_adj in mu_adjs]

        print(f"redistributing from {mu_0} to {mu_adjs} ({mu_adj_bins})")
        R_p[int(f"0b{mu_adj_bins[0][::-1]}", 2), int(f"0b{mu_adj_bins[+1][::-1]}", 2)] = 1
        R_m[int(f"0b{mu_adj_bins[0][::-1]}", 2), int(f"0b{mu_adj_bins[-1][::-1]}", 2)] = 1
    
    # Construct quantum circuit
    CR_p = UnitaryGate(R_p, label="$R_+$").control(3 + n_qubits_mask)
    CR_m = UnitaryGate(R_m, label="$R_-$").control(3 + n_qubits_mask)
    
    # Construct ancilla state preparation matrices via Householder reflection
    coefficients_AR = [
        angular_redistribution_coefficients[0], # I is applied to |00>
        angular_redistribution_coefficients[+1], # R_+ is applied to |01>
        angular_redistribution_coefficients[-1], # R_- is applied to |10>
        angular_redistribution_coefficients[0] # I is applied to |11> # @TEST - try shadow again with this instead of the 0
        # 0
    ]
    amplitudes_AR = np.sqrt(coefficients_AR / np.sum(coefficients_AR))
    print(amplitudes_AR)
    
    reflection_axis = np.zeros_like(amplitudes_AR)
    reflection_axis[0] = 1.0
    reflection_axis = reflection_axis - amplitudes_AR
    norm_squared = reflection_axis @ reflection_axis
    if np.isclose(norm_squared, 0.0):
        U_alpha_matrix = np.eye(len(amplitudes_AR)) - 2 * np.outer(reflection_axis, reflection_axis) / norm_squared
    else:
        U_alpha_matrix = np.eye(len(amplitudes_AR)) - 2 * np.outer(reflection_axis, reflection_axis) / norm_squared
    
    U_alpha = UnitaryGate(U_alpha_matrix, label="$U_\\text{alpha}$")
    
    qc = QuantumCircuit(qreg_mask, qreg_direction, qreg_switch, qreg_ancilla)
    
    qc.append(U_alpha, qreg_ancilla[ancilla_idxs_AR])

    # Flip switch so that controls act on intensity only
    qc.x(qreg_switch[:])
    
    # |s=0>|m=1>|a=01> -> CR_p
    qc.x(qreg_ancilla[ancilla_idxs_AR[0]])
    qc.append(CR_p, qreg_ancilla[ancilla_idxs_AR] + qreg_switch[:] + qreg_mask[:] + qreg_direction[:])
    qc.x(qreg_ancilla[ancilla_idxs_AR[0]])
    
    # |s=0>|m=1>|a=10> -> CR_m
    qc.x(qreg_ancilla[ancilla_idxs_AR[1]])
    qc.append(CR_m, qreg_ancilla[ancilla_idxs_AR] + qreg_switch[:] + qreg_mask[:] + qreg_direction[:])
    qc.x(qreg_ancilla[ancilla_idxs_AR[1]])
    
    qc.x(qreg_switch[:])
    
    qc.append(U_alpha, qreg_ancilla[ancilla_idxs_AR])
    
    return qc

def angular_redistribution_3d(
    m,
    delta_t,
    angular_redistribution_coefficients,
    idxs_dir, cs,
    adjacencies,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_mask, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_mask, qreg_direction, qreg_switch, qreg_ancilla,
    ancilla_idxs_AR,
):
    # Construct matrices
    dim_R = 2**(n_qubits_direction)
    
    edges = [
        (mu_0, mu_adj)
        for mu_0 in range(m)
        for mu_adj in range(mu_0+1, m)
        if adjacencies[mu_0][mu_adj] == 1
    ]

    matching_idxs = {}
    for edge in edges:
        mu_0, mu_adj = edge

        matching_idxs_incident = {
            matching_idx
            for edge_other, matching_idx in matching_idxs.items()
            if mu_0 in edge_other or mu_adj in edge_other
        }

        matching_idx = 0
        while matching_idx in matching_idxs_incident:
            matching_idx += 1

        matching_idxs[edge] = matching_idx
    
    n_matchings = max(matching_idxs.values()) + 1
    
    matchings = [
        [edge for edge, matching_idx in matching_idxs.items() if matching_idx == matching_idx_0]
        for matching_idx_0 in range(n_matchings)
    ]
    
    alpha = angular_redistribution_coefficients[0]
    
    W = alpha * np.eye(dim_R)
    
    for matching in matchings:
        R = np.eye(dim_R)
        
        for mu_0, mu_adj in matching:
            mu_0_bin = bin(mu_0)[2:].zfill(len(m_max_bin))
            mu_adj_bin = bin(mu_adj)[2:].zfill(len(m_max_bin))

            idx_0 = int(f"0b{mu_0_bin[::-1]}", 2)
            idx_adj = int(f"0b{mu_adj_bin[::-1]}", 2)

            R[idx_0, idx_0] = 0
            R[idx_adj, idx_adj] = 0
            R[idx_0, idx_adj] = 1
            R[idx_adj, idx_0] = 1
        
        W += (1 - alpha) / n_matchings * R
    
    print(f"redistributing over {n_matchings} matchings of the direction adjacency graph")
    
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    S = eigenvectors @ np.diag(np.sqrt(np.clip(1 - eigenvalues**2, 0.0, None))) @ eigenvectors.T
    
    U = np.block([[W, S], [S, -W]])
    ancilla_idx_AR = ancilla_idxs_AR[0]
    n_blocks = 2**(1 + n_qubits_switch + n_qubits_mask)
    
    W_conditioned = np.zeros((n_blocks * dim_R, n_blocks * dim_R))
    
    for block in range(n_blocks):
        ancilla = block % 2
        switch = (block >> 1) % 2
        mask = (block >> (1 + n_qubits_switch)) % 2 if n_qubits_mask > 0 else 1

        for ancilla_other in range(2):
            block_other = block - ancilla + ancilla_other

            if switch == 0 and mask == 1:
                W_conditioned[block::n_blocks, block_other::n_blocks] = U[
                    ancilla*dim_R:(ancilla+1)*dim_R,
                    ancilla_other*dim_R:(ancilla_other+1)*dim_R
                ]
            elif ancilla == ancilla_other:
                W_conditioned[block::n_blocks, block_other::n_blocks] = np.eye(dim_R)
    
    qc = QuantumCircuit(qreg_mask, qreg_direction, qreg_switch, qreg_ancilla)
    
    qc.append(
        UnitaryGate(W_conditioned, label="$W$"),
        [qreg_ancilla[ancilla_idx_AR]] + qreg_switch[:] + qreg_mask[:] + qreg_direction[:]
    )
    
    return qc

def special_boundary_conditions(
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    boundary_idxs,
    n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
    ancilla_idxs_BC,
    verbose=False
):
    ctrl_boundary = "1"
    ctrl_switch = "0"
    ctrl_state = ctrl_boundary + ctrl_switch

    qc = QuantumCircuit(qreg_boundary, qreg_switch, qreg_ancilla)

    absorb_gate = XGate().control(n_qubits_boundary + n_qubits_switch, ctrl_state=ctrl_state)
    qc.append(absorb_gate, qreg_switch[:] + qreg_boundary[:] + [qreg_ancilla[ancilla_idxs_BC[0]]])

    return qc

# Original propagation function (wraps single_direction_propagation and single_direction_propagation_angular_redistribution)
def propagation(
    n, m,
    N, M,
    idxs_dir, cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    with_mp=False,
    verbose=False
):
    # Prepare quantum circuit first since we need to construct it on the fly
    qc = QuantumCircuit(qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla)

    print("Available resources:", mp.cpu_count(), len(os.sched_getaffinity(0)), os.environ.get("SLURM_TASKS_PER_NODE"))
    if with_mp:
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = 1

    print(f"Computing propagation circuits using {cpu_count} CPUs", time.time())

    if cpu_count == 1:
        print(f"Running sequentially", time.time())

        results = [
            single_direction_propagation(
                mu,
                n, m,
                N, M,
                cs,
                idx_coord_map, coord_idx_map, m_max_bin,
                n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
                qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
                None,
                verbose
            )
            for mu in range(m)
        ]

        # Unzip results
        single_direction_propagation_gates = [result[0] for result in results]
        single_direction_propagation_gate_qubits = [result[1] for result in results]
    else:
        # Use multiprocessing pool
        with Pool(cpu_count) as pool:
            print(f"Opened multiprocessing pool", time.time())

            results = pool.starmap(
                single_direction_propagation,
                [
                    (
                        mu,
                        n, m,
                        N, M,
                        cs,
                        idx_coord_map, coord_idx_map, m_max_bin,
                        n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
                        qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
                        None,
                        verbose
                    )
                    for mu in range(m)
                ]
            )

            # Unzip results
            single_direction_propagation_gates = [result[0] for result in results]
            single_direction_propagation_gate_qubits = [result[1] for result in results]

    return single_direction_propagation_gates, single_direction_propagation_gate_qubits

# Called by propagation - processes a single direction
def single_direction_propagation(
    mu,
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    cache_option=None,
    verbose=False
):
    print(f"Current process {os.getpid()} working on direction {mu}", time.time())

    if isinstance(cache_option, str):
        cache_option = cache_option.lower()
        if cache_option not in ["qpy", "pkl"]:
            raise ValueError(f"Invalid cache_option '{cache_option}' - must be one of None (default), 'qpy', or 'pkl'.")
    elif cache_option is not None:
        raise TypeError(f"Invalid cache_option {cache_option} of type {type(cache_option)} - must be one of None (default), 'qpy', or 'pkl'.")

    try:
        if cache_option is None:
            raise Exception("User requested cache_option=None, skipping cache load attempt")

        print("Trying to open propagation circuit file...", time.time())

        if n == 1:
            f = open(f".cache/CP_mu_circuit_D{n}Q{m}_{N[0]}_{mu}.{cache_option}", "rb")
        else:
            N_str = "-".join([str(N_i) for N_i in N])
            f = open(f".cache/CP_mu_circuit_D{n}Q{m}_{N_str}_{mu}.{cache_option}", "rb")

        print("Propagation circuit file detected...", time.time())

        if cache_option == "qpy":
            CP_mu_circuit = qpy.load(f)[0]
        else:
            CP_mu_circuit = pickle.load(f)

        CP_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

        print(f"Successfully loaded propagation circuit for direction {mu} for a D{n}Q{m} scheme on lattice w/ shape {N} from cache", time.time())

        f.close()

        print(f"Closed file", time.time())
    except Exception as e:
        print(f"Could not load propagation circuit for direction {mu} for a D{n}Q{m} scheme on lattice w/ shape {N} from cache due to error '{e}'; computing...", time.time())

        mu_bin = bin(mu)[2:].zfill(len(m_max_bin))
        if verbose:
            print(f"Process {os.getpid().name} processing direction {mu} (binary {mu_bin})")

        c_mu = cs[mu]

        switch_bin = "0"

        ctrl_direction = mu_bin[::-1]
        ctrl_switch = switch_bin
        ctrl_state = ctrl_direction + ctrl_switch

        print("beginning P_mu circuit computation", time.time())
        
        P_mu_circuit = QuantumCircuit(qreg_lattice)

        n_qubits_axis_offset = 0
        for axis in range(n):
            n_qubits_axis = int(round(np.log2(N[axis])))

            if 2**n_qubits_axis != N[axis]:
                raise ValueError(f"Axis resolution N[{axis}]={N[axis]} is not a power of 2, which is required for arithmetic propagation.")
            
            qreg_axis = qreg_lattice[n_qubits_axis_offset:n_qubits_axis_offset+n_qubits_axis][::-1]
            
            displacement = -(c_mu[axis] if n > 1 else c_mu)
            
            if verbose:
                print(f"Translating axis {axis} ({n_qubits_axis} qubits) by {displacement} for direction {mu}")

            single_axis_propagation(P_mu_circuit, qreg_axis, displacement)

            n_qubits_axis_offset += n_qubits_axis

        print("beginning CP_mu circuit computation", time.time())

        # CP_mu_circuit acts on the lattice qubits, but appropriately controlled by the direction and switch qubits
        CP_mu_circuit = P_mu_circuit.control(n_qubits_direction + n_qubits_switch, ctrl_state=ctrl_state, label=f"$P_{mu}$")
        CP_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

        print("saving propagation circuit", time.time())

        if n == 1:
            f = open(f".cache/CP_mu_circuit_D{n}Q{m}_{N[0]}_{mu}.{cache_option}", "rb")
        else:
            N_str = "-".join([str(N_i) for N_i in N])
            f = open(f".cache/CP_mu_circuit_D{n}Q{m}_{N_str}_{mu}.{cache_option}", "wb")

        if cache_option == "qpy":
            qpy.dump(CP_mu_circuit, f)
        else:
            pickle.dump(CP_mu_circuit, f, protocol=5)

        print(f"Successfully computed propagation circuit for direction {mu} for a D{n}Q{m} scheme on lattice w/ shape {N} and saved to cache")

        del P_mu_circuit

        f.close()
    finally:
        print(f"Returning", time.time())

        return CP_mu_circuit, CP_mu_qubits

def single_axis_propagation(P_mu_circuit, qreg_axis, displacement):
    for bit in range(len(qreg_axis)):
        if not (abs(displacement) >> bit) & 1:
            continue

        qreg_field = qreg_axis[bit:]
        n_qubits_field = len(qreg_field)

        if displacement > 0:
            for q in range(n_qubits_field-1, 0, -1):
                P_mu_circuit.mcx(qreg_field[0:q], qreg_field[q])

            P_mu_circuit.x(qreg_field[0])
        elif displacement < 0:
            P_mu_circuit.x(qreg_field[0])

            for q in range(1, n_qubits_field):
                P_mu_circuit.mcx(qreg_field[0:q], qreg_field[q])
