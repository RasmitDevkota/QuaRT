import numpy as np

import multiprocessing as mp
import time

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, XGate, ZGate

def state_preparation(
    I_i, S_i,
    m,
    delta_t,
    coord_idx_map, m_max_bin,
    n_qubits,
    qreg_head, qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
    verbose=False
):
    initial_statevector = np.zeros((2**n_qubits))

    anc_bin = "000"

    prob_amp_I = 0
    prob_amp_S = 0

    for c, (coordinate_idx, coordinate_bin) in enumerate(coord_idx_map.items()):
        coordinate_bin = coordinate_bin[::-1]

        for mu in range(m):
            mu_bin = bin(mu)[2:].zfill(len(m_max_bin))#[::-1]

            for s_bin in range(2):
                if s_bin == 0:
                    prob_amp = I_i[c, mu]
                    prob_amp_I += prob_amp
                else:
                    prob_amp = 0.5 * delta_t * S_i[c, mu]
                    # @TODO - why does this fix it, I know it effectively has to do with normalization,
                    #         but I think it should be performed somewhere else
                    prob_amp *= 2
                    prob_amp_S += prob_amp

                idx_bin = f"0b{anc_bin}{s_bin}{mu_bin}{coordinate_bin}0"
                idx_dec = int(idx_bin, 2)
                initial_statevector[idx_dec] = prob_amp

    norm = np.linalg.norm(initial_statevector)
    if verbose:
        print(norm)
    if norm > 0:
        initial_statevector /= norm

    print(prob_amp_I, prob_amp_S)

    qc = QuantumCircuit(qreg_head, qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla)
    qc.initialize(initial_statevector)

    return qc

def absorption_scattering(
    kappa,
    delta_t,
    I_2,
    n_qubits_direction,
    qreg_direction, qreg_switch, qreg_ancilla
):
    # Original matrix
    A = (1 - kappa * delta_t) * I_2

    # LCU method
    a = 1 - kappa * delta_t
    b = 2 * np.sqrt(1 - a**2)

    C_1 = np.array([
        [a + 0.5j * b, 0],
        [0, a + 0.5j * b]
    ])

    C_2 = np.array([
        [a - 0.5j * b, 0],
        [0, a - 0.5j * b]
    ])

    # Check that LCU method recovers the original matrix
    assert np.sum(A - 0.5*(C_1 + C_2)) == 0j

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

def absorption_emission(I_2M, Z_2M, qreg_switch, qreg_ancilla):
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

def propagation(
    n, m,
    N, M,
    idxs_dir, cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    angular_redistribution_coefficients,
    n_qubits_direction, n_qubits_switch, n_qubits_lattice,
    qreg_lattice, qreg_direction, qreg_switch,
    verbose=False
):
    # Prepare quantum circuit first since we need to construct it on the fly
    qc = QuantumCircuit(qreg_lattice, qreg_direction, qreg_switch)

    cpu_count = mp.cpu_count()
    batch_size = max(int(np.ceil(m/cpu_count)), 1)

    # If there are more than one non-zero entries, we use angular redistribution
    with_angular_redistribution = False
    if angular_redistribution_coefficients is not None:
        if any([np.count_nonzero(angular_redistribution_coefficients[mu]) > 1 for mu in range(m)]):
            with_angular_redistribution = True

            print("Propagating with angular redistribution")

    pool = mp.Pool(cpu_count)
    if with_angular_redistribution:
        results = pool.starmap(
            single_direction_propagation_angular_redistribution,
            [
                (
                    mu,
                    n, m,
                    N, M,
                    idxs_dir, cs,
                    idx_coord_map, coord_idx_map, m_max_bin,
                    angular_redistribution_coefficients,
                    n_qubits_direction, n_qubits_switch, n_qubits_lattice,
                    qreg_lattice, qreg_direction, qreg_switch,
                    verbose
                )
                for mu in range(0, m, batch_size)
            ]
        )
    else:
        results = pool.starmap(
            single_direction_propagation,
            [
                (
                    mu,
                    n, m,
                    N, M,
                    cs,
                    idx_coord_map, coord_idx_map, m_max_bin,
                    n_qubits_direction, n_qubits_switch,
                    qreg_lattice, qreg_direction, qreg_switch,
                    verbose
                )
                for mu in range(0, m, batch_size)
            ]
        )

    single_direction_propagation_gates = [result[0] for result in results]
    single_direction_propagation_gate_qubits = [result[1] for result in results]
    # print(single_direction_propagation_gates)
    pool.close()

    for mu in range(m):
        qc.append(single_direction_propagation_gates[mu], single_direction_propagation_gate_qubits[mu])

    return qc

def single_direction_propagation(
    mu,
    n, m,
    N, M,
    cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    n_qubits_direction, n_qubits_switch,
    qreg_lattice, qreg_direction, qreg_switch,
    verbose=False
):
    P_mu = np.zeros((M, M))

    mu_bin = bin(mu)[2:].zfill(len(m_max_bin))
    if verbose:
        print(f"Process {mp.current_process().name} processing direction {mu} (binary {mu_bin})")

    c_mu = cs[mu]

    for idx_init_bin, coord_init in idx_coord_map.items():
        # Periodic BCs
        if n == 1:
            coord_dest = ((coord_init[0] + c_mu) % N[0],)
        else:
            coord_dest = tuple((coord_init[i] + c_mu[i]) % N[i] for i in range(n))

        idx_dest_bin = coord_idx_map[coord_dest]

        idx_init_dec = int(f"0b{idx_init_bin[::-1]}", 2)
        idx_dest_dec = int(f"0b{idx_dest_bin[::-1]}", 2)

        if verbose:
            print(coord_init, coord_dest)
            print(idx_init_bin, idx_dest_bin)
            print(idx_init_dec, idx_dest_dec)

        P_mu[idx_init_dec, idx_dest_dec] = 1

    # @TODO - verify byte-Endianness on the ctrl_state - it seems to not matter?
    ctrl_switch = "0"
    ctrl_direction = mu_bin
    ctrl_state = ctrl_direction + ctrl_switch
    print(ctrl_state)

    print("---")

    print(time.time())

    # P_mu_circuit only acts on the lattice qubits
    P_mu_circuit = QuantumCircuit(qreg_lattice)
    P_mu_circuit.unitary(P_mu, qreg_lattice[:])

    # CP_mu_circuit acts on the lattice qubits, but appropriately controlled by the direction and switch qubits
    CP_mu_circuit = P_mu_circuit.control(n_qubits_direction + n_qubits_switch, ctrl_state=ctrl_state)
    CP_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

    print(time.time())

    return CP_mu_circuit, CP_mu_qubits

def single_direction_propagation_angular_redistribution(
    mu0,
    n, m,
    N, M,
    idxs_dir, cs,
    idx_coord_map, coord_idx_map, m_max_bin,
    angular_redistribution_coefficients,
    n_qubits_direction, n_qubits_switch, n_qubits_lattice,
    qreg_lattice, qreg_direction, qreg_switch,
    verbose=False
):
    print("="*80)

    dim_P_mu = 2**(n_qubits_lattice + n_qubits_direction)
    P_mu = np.zeros((dim_P_mu, dim_P_mu))

    mu0_idx = idxs_dir[mu0]
    mu0_bin = bin(mu0)[2:].zfill(len(m_max_bin))
    c_mu = cs[mu0]
    if verbose:
        print(f"Process {mp.current_process().name} processing direction {mu0} (binary {mu0_bin})")

    # The adjacent indices can be accessed using integer deltas
    match n:
        case 1:
            idx_deltas = [0]

            mu_adj_idxs = [mu0_idx]
        case 2:
            idx_deltas = [+1, 0, -1]

            mu_adj_idxs = [(mu0_idx + idx_delta) % m for idx_delta in idx_deltas]
        case 3:
            # @TODO - determine which is better (first is easier to use, second may be more accurate)
            idx_deltas = [(1,0), (0,1), (-1,0), (0,-1)]
            # idx_deltas = [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)]

            mu_adj_idxs = [(list(np.array(mu0_idx) + np.array(idx_delta))) % m for idx_delta in idx_deltas]

    mu_adjs = [idxs_dir[mu_adj_idx] for mu_adj_idx in mu_adj_idxs]
    mu_adj_bins = [bin(mu_adj)[2:].zfill(len(m_max_bin)) for mu_adj in mu_adjs]
    c_adjs = [cs[mu_adj] for mu_adj in mu_adjs]

    ctrl_switch = "0"
    ctrl_direction = mu0_bin

    # Loop through each initial coordinate
    for idx_init_bin, coord_init in idx_coord_map.items():
        # Compute destination coordinate using periodic BCs
        if n == 1:
            coord_dest = ((coord_init[0] + c_mu) % N[0],)
        else:
            coord_dest = tuple((coord_init[i] + c_mu[i]) % N[i] for i in range(n))

        # Compute binary string representation of destination coordinate index
        idx_dest_bin = coord_idx_map[coord_dest]

        # Loop through adjacent directions
        for mu_adj, mu_adj_bin in zip(mu_adjs, mu_adj_bins):
            # Get the full binary string representation of the initial and final coordinate+direction state
            full_init_bin = idx_init_bin + ctrl_direction# + ctrl_switch
            full_dest_bin = idx_dest_bin + mu_adj_bin# + ctrl_switch

            # Get the decimal value of the above binary strings - these are the matrix indices in P_mu
            full_init_dec = int(f"0b{full_init_bin[::-1]}", 2)
            full_dest_dec = int(f"0b{full_dest_bin[::-1]}", 2)

            # if verbose:
            #     print(coord_init, coord_dest)
            #     print(mu_adj_bin, full_init_bin, full_dest_bin)
            #     print(mu_adj_bin, full_init_dec, full_dest_dec)

            # Set entry of P_mu to corresponding angular redistribution coefficient
            P_mu[full_init_dec, full_dest_dec] = angular_redistribution_coefficients[mu0][mu_adj]

            print(f"redistributing {angular_redistribution_coefficients[mu0][mu_adj]} at {coord_init} from direction {mu0} to {mu_adj} {mu_adj_bin},", full_init_bin, full_dest_bin)

            print("-")

        print("-"*20)

    print(dim_P_mu, P_mu)

    for i in range(dim_P_mu):
        colsum = np.count_nonzero(P_mu[:,i])
        rowsum = np.count_nonzero(P_mu[i,:])

        print("col", i, "has sum", colsum)
        print("row", i, "has sum", rowsum)

        # if colsum == 0 and rowsum == 0:
        print(bin(i))

        print("===")

    print("---")

    print(time.time())

    # # P_mu_circuit acts on the lattice qubits, but appropriately controlled by the direction and switch qubits
    # P_mu_circuit = QuantumCircuit(n_qubits_lattice, n_qubits_direction + n_qubits_switch)
    # P_mu_circuit.unitary(P_mu, qreg_lattice[:])
    #
    # P_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]
    #
    # print(time.time())
    #
    # return P_mu_circuit, P_mu_qubits

    # P_mu_circuit only acts on the lattice and direction qubits
    P_mu_circuit = QuantumCircuit(qreg_lattice, qreg_direction)
    P_mu_circuit.unitary(P_mu, qreg_direction[:] + qreg_lattice[:])

    # CP_mu_circuit acts on the lattice and direction qubits, but appropriately controlled by the switch qubits
    ctrl_state = "0"
    CP_mu_circuit = P_mu_circuit.control(n_qubits_switch, ctrl_state=ctrl_state)
    CP_mu_qubits = qreg_switch[:] + qreg_direction[:] + qreg_lattice[:]

    print(time.time())

    return CP_mu_circuit, CP_mu_qubits

