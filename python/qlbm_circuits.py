import numpy as np
np.set_printoptions(threshold=np.inf)

import multiprocessing as mp
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, XGate, ZGate
from qiskit.quantum_info import SparsePauliOp

import pennylane as qml

def state_preparation(
    I_i, S_i,
    n, m,
    N,
    delta_t,
    coord_idx_map, m_max_bin,
    boundary_idxs, boundary_conditions,
    n_qubits, n_qubits_ancilla,
    qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    initial_statevector = np.zeros((2**n_qubits))

    anc_bin = "0"*n_qubits_ancilla

    prob_amp_I = 0
    prob_amp_S = 0

    norm_I = np.max(I_i) #np.linalg.norm(I_i)

    norm_S = np.max(S_i) #np.linalg.norm(S_i)
    if norm_I > 0 and norm_S > 0:
        S_i = S_i * norm_I/norm_S

    print("actually recovering intensities and sources:")
    print(I_i, S_i)

    for c, (coordinate, coordinate_bin) in enumerate(coord_idx_map.items()):
        coordinate_bin = coordinate_bin[::-1]

        for mu in range(m):
            mu_bin = bin(mu)[2:].zfill(len(m_max_bin))[::-1]

            for s_bin in range(2):
                if s_bin == 0:
                    match n:
                        case 1:
                            boundary_bin = "1" if (
                                (coordinate[0] == 0 and mu in boundary_idxs[0] and boundary_conditions[0] == "absorb") or
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
                            boundary_bin = "0"
                            boundary_bin = "1" if (
                                (coordinate[0] == 0 and mu in boundary_idxs[0] and boundary_conditions[0][0] == "absorb") or
                                (coordinate[0] == N[0]-1 and mu in boundary_idxs[1] and boundary_conditions[1][0] == "absorb") or
                                (coordinate[1] == 0 and mu in boundary_idxs[2] and boundary_conditions[2][0] == "absorb") or
                                (coordinate[1] == N[1]-1 and mu in boundary_idxs[3] and boundary_conditions[3][0] == "absorb")
                                (coordinate[2] == 0 and mu in boundary_idxs[4] and boundary_conditions[4][0] == "absorb") or
                                (coordinate[2] == N[2]-1 and mu in boundary_idxs[5] and boundary_conditions[5][0] == "absorb")
                            ) else "0"

                    if boundary_bin[0] == "1":
                        print("Absorb boundary conditions @", coordinate)

                    prob_amp = I_i[c, mu]
                    prob_amp_I += prob_amp
                else:
                    prob_amp = 0.5 * delta_t * S_i[c, mu]
                    # @TODO - why does this kind of fix it? I know it effectively has to do with normalization,
                    #         but I think it should be performed somewhere else
                    prob_amp *= 2.0
                    prob_amp_S += prob_amp

                idx_bin = f"0b{anc_bin}{s_bin}{mu_bin}{boundary_bin}{coordinate_bin}"
                idx_dec = int(idx_bin, 2)
                initial_statevector[idx_dec] = prob_amp

    norm = np.linalg.norm(initial_statevector)
    if verbose:
        print(norm)
    if norm > 0:
        initial_statevector /= norm

    print("prob_amps:", prob_amp_I, prob_amp_S)

    qc = QuantumCircuit(qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla)
    qc.initialize(initial_statevector)

    return qc

def absorption_scattering(
    kappa, sigma,
    delta_t,
    I_2,
    n_qubits_direction,
    qreg_direction, qreg_switch, qreg_ancilla
):
    # Hardcoded parameters for testing
    kappa = 0.0
    sigma = 0.0
    delta_t = 1

    # LCU method
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

    qc.h(qreg_ancilla[0])

    for q in range(n_qubits_direction):
        qc.x(qreg_switch[:])

        qc.x(qreg_ancilla[0])
        qc.append(C_1_gate, [qreg_ancilla[0]] + qreg_switch[:] + [qreg_direction[q]])
        qc.x(qreg_ancilla[0])

        qc.append(C_2_gate, [qreg_ancilla[0]] + qreg_switch[:] + [qreg_direction[q]])
        qc.x(qreg_switch[:])

    qc.h(qreg_ancilla[0])

    return qc

def absorption_emission(
    I_2M, Z_2M,
    qreg_switch, qreg_ancilla
):
    # Original matrix
    B = np.block([
        [I_2M, I_2M],
        [Z_2M, I_2M]
    ])

    # LCU method
    D_1 = XGate()
    D_2 = XGate()
    D_3 = ZGate()

    # Check that LCU method recovers the original matrix
    assert np.sum(B - np.kron((D_1.to_matrix() + 0.5 * D_2.to_matrix() + 0.5 * D_3.to_matrix()), I_2M)) == 0j

    # Construct quantum operations
    D_1_gate = UnitaryGate(D_1, label="$D_1$").control(2)
    D_2_gate = UnitaryGate(D_2, label="$D_2$").control(2)
    D_3_gate = UnitaryGate(D_3, label="$D_3$").control(2)

    # Perform quantum circuit composition steps
    qc = QuantumCircuit(qreg_switch, qreg_ancilla)

    qc.h(qreg_ancilla[1])
    qc.h(qreg_ancilla[2])

    qc.x(qreg_ancilla[2])
    qc.append(D_1_gate, qreg_ancilla[1:3] + qreg_switch[:])
    qc.x(qreg_ancilla[2])

    qc.append(D_2_gate, qreg_ancilla[1:3] + qreg_switch[:])

    qc.append(D_3_gate, qreg_ancilla[1:3] + qreg_switch[:])

    qc.h(qreg_ancilla[1])
    qc.h(qreg_ancilla[2])

    return qc

def angular_redistribution(
    m,
    delta_t,
    angular_redistribution_coefficients,
    idxs_dir, cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_direction, qreg_switch, qreg_ancilla
):
    # if isinstance(angular_redistribution_coefficients, int):
    #     raise NotImplementedError("Global angular redistribution coefficient input is currently not implemented. Please provide an array of angular redistribution coefficients!")
    # elif not hasattr(angular_redistribution_coefficients, "__iter__"):
    #     raise ValueError("Please provide an array of angular redistribution coefficients!")

    # Construct matrices
    dim_R = 2**(n_qubits_direction)

    R_0 = np.eye((dim_R))
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

    print(R_0)
    print(R_p)
    print(R_m)

    # Construct quantum circuit
    CR_0 = UnitaryGate(R_0, label="$R_0$").control(3)
    CR_p = UnitaryGate(R_p, label="$R_+$").control(3)
    CR_m = UnitaryGate(R_m, label="$R_-$").control(3)

    qc = QuantumCircuit(qreg_direction, qreg_switch, qreg_ancilla)

    qc.h(qreg_ancilla[3:5])
    qc.x(qreg_switch[:])

    # 000 -> CR_0
    qc.append(CR_0, qreg_ancilla[3:5] + qreg_switch[:] + qreg_direction[:])

    # 011 -> CR_0
    qc.x(qreg_ancilla[3:5])
    qc.append(CR_0, qreg_ancilla[3:5] + qreg_switch[:] + qreg_direction[:])
    qc.x(qreg_ancilla[3:5])

    # 001 -> CR_p
    qc.x(qreg_ancilla[3])
    qc.append(CR_p, qreg_ancilla[3:5] + qreg_switch[:] + qreg_direction[:])
    qc.x(qreg_ancilla[3])

    # 010 -> CR_m
    qc.x(qreg_ancilla[4])
    qc.append(CR_m, qreg_ancilla[3:5] + qreg_switch[:] + qreg_direction[:])
    qc.x(qreg_ancilla[4])

    qc.x(qreg_switch[:])
    qc.h(qreg_ancilla[3:5])

    return qc

def apply_boundary_conditions(
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    boundary_idxs,
    n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    ctrl_boundary = "1"
    ctrl_switch = "0"
    ctrl_state = ctrl_boundary + ctrl_switch

    qc = QuantumCircuit(qreg_boundary, qreg_switch, qreg_ancilla)

    absorb_gate = XGate().control(n_qubits_boundary + n_qubits_switch, ctrl_state=ctrl_state)
    qc.append(absorb_gate, qreg_switch[:] + qreg_boundary[:] + [qreg_ancilla[5]])

    return qc

# Original propagation function (wraps single_direction_propagation and single_direction_propagation_angular_redistribution)
def propagation(
    n, m,
    N, M,
    idxs_dir, cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    angular_redistribution_coefficients,
    boundary_idxs, boundary_conditions,
    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    # Prepare quantum circuit first since we need to construct it on the fly
    qc = QuantumCircuit(qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla)

    cpu_count = mp.cpu_count()
    batch_size = max(int(np.ceil(m/cpu_count)), 1)
    print(cpu_count, batch_size)

    if boundary_conditions is None:
        boundary_conditions = [("periodic", None)]*4
    else:
        if len(boundary_conditions) == 4:
            if len(set(boundary_conditions)) != 1:
                raise ValueError("Boundary conditions currently must be the same for all walls!")
        elif len(boundary_conditions) != 4:
            raise ValueError("Boundary conditions must be specified for all walls!")

    # Use multiprocessing pool
    with mp.Pool(cpu_count) as pool:
        # results = [_propagation(
        #     n, m,
        #     N, M,
        #     cs,
        #     idx_coord_map, coord_idx_map, m_max_bin,
        #     boundary_idxs, boundary_conditions,
        #     n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
        #     qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
        #     verbose
        # )]
        results = pool.starmap(
            single_direction_propagation,
            [
                (
                    mu,
                    n, m,
                    N, M,
                    cs,
                    idx_coord_map, coord_idx_map, m_max_bin,
                    boundary_idxs, boundary_conditions,
                    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
                    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
                    verbose
                )
                for mu in range(m)
            ],
            chunksize=batch_size
        )

        # Unzip results
        single_direction_propagation_gates = [result[0] for result in results]
        single_direction_propagation_gate_qubits = [result[1] for result in results]

    for gate, qubits in zip(single_direction_propagation_gates, single_direction_propagation_gate_qubits):
        qc.append(gate, qubits)

    return qc

# Called by propagation - processes all directions at once
def _propagation(
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    boundary_idxs, boundary_conditions,
    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    dim_P = 2**(n_qubits_lattice + n_qubits_direction + n_qubits_switch + n_qubits_ancilla)
    P = np.zeros((dim_P, dim_P))

    switch_bin = "0"
    ancilla_init_bin = "0"*n_qubits_ancilla

    for mu in range(m):
        mu_bin = bin(mu)[2:].zfill(len(m_max_bin))
        if verbose:
            print(f"Process {mp.current_process().name} processing direction {mu} (binary {mu_bin})")

        c_mu = cs[mu]

        for idx_init_bin, coord_init in idx_coord_map.items():
            coord_dest = None

            # Currently, non-periodic boundary conditions are only supported for 2D+ lattices
            if n >= 2:
                if all([boundary_conditions[i][0] == "absorb" for i in range(4)]) and \
                (
                    coord_init[0] == 0 and mu in boundary_idxs[0] or
                    coord_init[0] == N[0]-1 and mu in boundary_idxs[1] or
                    coord_init[1] == 0 and mu in boundary_idxs[2] or
                    coord_init[1] == N[1]-1 and mu in boundary_idxs[3]
                ):
                    print("Using absorb BCs on all walls")
                    coord_dest = coord_init
                    # ancilla_dest_bin = ancilla_init_bin
                    ancilla_dest_bin = "0"*(n_qubits_ancilla-1) + "1"

            # If the required variables are not set (so none of the above cases were met), use periodic boundary conditions by default
            if coord_dest is None or ancilla_dest_bin is None:
                print("Using periodic BCs on all walls")
                if n == 1:
                    coord_dest = ((coord_init[0] + c_mu) % N[0],)
                else:
                    coord_dest = tuple((coord_init[i] + c_mu[i]) % N[i] for i in range(n))

                ancilla_dest_bin = ancilla_init_bin

            idx_dest_bin = coord_idx_map[coord_dest]

            full_init_bin = idx_init_bin + mu_bin + switch_bin + ancilla_init_bin
            full_dest_bin = idx_dest_bin + mu_bin + switch_bin + ancilla_dest_bin

            full_init_dec = int(f"0b{full_init_bin[::-1]}", 2)
            full_dest_dec = int(f"0b{full_dest_bin[::-1]}", 2)

            if verbose:
                print(f"Propagating from coordinate {coord_init} to {coord_dest} == from binary {full_init_bin} to binary {full_dest_bin} == from index {full_init_dec} to index {full_dest_dec}")

            P[full_init_dec, full_dest_dec] = 1

            ###################################
            for a in range(int(2**n_qubits_ancilla)):
                coord_dest = tuple((coord_init[i] + c_mu[i]) % N[i] for i in range(n))
                idx_dest_bin = coord_idx_map[coord_dest]
                full_init_bin = idx_init_bin + mu_bin + switch_bin + ancilla_init_bin
                full_dest_bin = idx_dest_bin + mu_bin + switch_bin + ancilla_init_bin

                full_init_dec = int(f"0b{full_init_bin[::-1]}", 2)
                full_dest_dec = int(f"0b{full_dest_bin[::-1]}", 2)
                P[full_init_dec, full_dest_dec] = 1
            ###################################

    print("before:")
    print(np.sum(P, axis=0))
    print(np.sum(P, axis=1))

    # P[dim_P//32:, dim_P//32:] = np.eye(31*dim_P//32)

    print("after:")
    print(np.sum(P, axis=0))
    print(np.sum(P, axis=1))

    # print(P)
    print(dim_P, dim_P//32)
    print(P[:dim_P//32, :dim_P//32])

    print("---")

    print(time.time())

    P_circuit = QuantumCircuit(qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla, label="$P$")
    P_circuit.unitary(P, qreg_ancilla[:] + qreg_switch[:] + qreg_direction[:] + qreg_lattice[:])

    P_qubits = qreg_ancilla[:] + qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

    print(time.time())

    return P_circuit, P_qubits

# Called by propagation - processes a single direction
def single_direction_propagation(
    mu,
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    boundary_idxs, boundary_conditions,
    n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    dim_P_mu = 2**(n_qubits_lattice)
    P_mu = np.zeros((dim_P_mu, dim_P_mu))

    mu_bin = bin(mu)[2:].zfill(len(m_max_bin))
    if verbose:
        print(f"Process {mp.current_process().name} processing direction {mu} (binary {mu_bin})")

    c_mu = cs[mu]

    switch_bin = "0"

    ctrl_direction = mu_bin[::-1]
    ctrl_switch = switch_bin
    ctrl_state = ctrl_direction + ctrl_switch

    for idx_init_bin, coord_init in idx_coord_map.items():
        # Use periodic boundary conditions
        if n == 1:
            coord_dest = ((coord_init[0] + c_mu) % N[0],)
        else:
            coord_dest = tuple((coord_init[i] + c_mu[i]) % N[i] for i in range(n))

        idx_dest_bin = coord_idx_map[coord_dest]

        idx_init_dec = int(f"0b{idx_init_bin[::-1]}", 2)
        idx_dest_dec = int(f"0b{idx_dest_bin[::-1]}", 2)

        if verbose:
            print(f"Propagating from coordinate {coord_init} to {coord_dest} == from binary {idx_init_bin} to binary {idx_dest_bin} == from index {idx_init_dec} to index {idx_dest_dec}")

        P_mu[idx_init_dec, idx_dest_dec] = 1

    print(P_mu)

    print("---")

    print(time.time())

    # P_mu_circuit only acts on the lattice qubits
    P_mu_circuit = QuantumCircuit(qreg_lattice)
    P_mu_circuit.unitary(P_mu, qreg_lattice[:])

    print(time.time())

    # CP_mu_circuit acts on the lattice qubits, but appropriately controlled by the direction and switch qubits
    CP_mu_circuit = P_mu_circuit.control(n_qubits_direction + n_qubits_switch, ctrl_state=ctrl_state, label=f"$P_{mu}$")
    CP_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

    print(time.time())

    return CP_mu_circuit, CP_mu_qubits

