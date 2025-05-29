import numpy as np
np.set_printoptions(linewidth=10000)

import matplotlib.pyplot as plt

import time

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from statevector_to_str import statevector_to_str

from qlbm_circuits import state_preparation, absorption_scattering, absorption_emission, angular_redistribution, propagation
from qlbm_utils import compute_memory_requirements, allocate_registers, compute_binary_representations, map_indices_coordinates, construct_identity_matrices
from lbm_utils import compute_grid_parameters, compute_scheme_velocities, compute_scheme_boundaries
from analysis import measurements_to_lattice, statevector_analysis, statevector_analysis_deep

def simulate(
    I_i, S_i,
    n, m,
    N,
    n_it=1,
    delta_t=1,
    kappa=0.0,
    sigma=0.0,
    angular_redistribution_coefficients=None,
    boundary_conditions=None,
    save_lattices=False,
    save_circuit=False,
    statevector_analysis_options=None
):
    time_start = time.time()

    # 0. Input parsing
    if statevector_analysis_options is None:
        statevector_analysis_options = []

    # 1. Lattice utilities
    M_0, M = compute_grid_parameters(n, N)
    idxs_dir, cs = compute_scheme_velocities(n, m)
    boundary_idxs = compute_scheme_boundaries(n, m)
    coordinate_max_bin, m_max_bin = compute_binary_representations(m, M)
    idx_coord_map, coord_idx_map = map_indices_coordinates(N, coordinate_max_bin)

    # 2. Initial conditions, sources, boundaries
    if I_i is None:
        I_i = np.zeros(shape=(M, m))

    if S_i is None:
        S_i = np.zeros(shape=(M, m))

    # 3. Quantum system construction
    n_qubits, n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla = compute_memory_requirements(m, M_0)

    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla, creg_measure = allocate_registers(n_qubits, n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla)

    I_2, I_M, I_2M, I_4M, Z_2M = construct_identity_matrices(M)

    # Absorption-Scattering, Absorption-Emission, Propagation, and Angular Redistribution circuits are constant for time-independent simulation setups
    ASCircuit = absorption_scattering(
        kappa, sigma,
        delta_t,
        I_2,
        n_qubits_direction,
        qreg_direction, qreg_switch, qreg_ancilla
    )
    print("Absorption-Scattering circuit breakdown:", ASCircuit.count_ops())

    # If there are more than one non-zero entries, we use angular redistribution
    with_angular_redistribution = True or angular_redistribution_coefficients is not None
    if with_angular_redistribution:
        print("Applying angular redistribution")

        ARCircuit = angular_redistribution(
            m,
            delta_t,
            angular_redistribution_coefficients,
            idxs_dir, cs,
            idx_coord_map, coord_idx_map, m_max_bin,
            n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
            qreg_direction, qreg_switch, qreg_ancilla
        )
        print("Angular Redistribution circuit breakdown:", ASCircuit.count_ops())

    AECircuit = absorption_emission(
        I_2M, Z_2M,
        qreg_switch, qreg_ancilla
    )
    print("Absorption-Emission circuit breakdown:", AECircuit.count_ops())

    PCircuit = propagation(
        n, m,
        N, M,
        idxs_dir, cs,
        idx_coord_map, coord_idx_map, m_max_bin,
        angular_redistribution_coefficients,
        boundary_idxs, boundary_conditions,
        n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
        qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
        verbose=False#True
    )
    print("Propagation circuit breakdown:", PCircuit.count_ops())

    qc_0 = QuantumCircuit(
        qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
        creg_measure
    )

    # Log qubit usage
    lattice_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_lattice[:]]
    auxiliary_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:]]
    direction_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_direction[:]]
    switch_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_switch[:]]
    ancilla_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_ancilla[:]]
    print(
        f"Lattice qubits: {lattice_qubits}",
        f"Auxiliary qubits: {auxiliary_qubits}",
        f"Direction qubits: {direction_qubits}",
        f"Switch qubits: {switch_qubits}",
        f"Ancilla qubits: {ancilla_qubits}",
        sep="\n"
    )

    # 4. Simulation
    qc_prev = qc_0
    I_prev, S_prev = np.copy(I_i), np.copy(S_i)

    lattices = []

    # Set up AerSimulator in advance (we only need to do this once)
    if "GPU" in AerSimulator().available_devices():
        # If a GPU is available, prefer cuQuantum's cuStateVec library to accelerate simulation (if built with support)
        backend = AerSimulator(
            method="automatic",
            device="GPU",
            cuStateVec_enable=True
        )
        print("Using GPU")
    else:
        backend = AerSimulator(
            method="automatic",
            max_parallel_threads=16,
            max_parallel_shots=1,
            max_parallel_experiments=1
        )
        print("Using CPU")

    # Simulate
    for it in range(n_it+1):
        print(f"========== Iteration {it} ==========")

        qc = qc_0
        # qc = qc_prev

        if "initial" in statevector_analysis_options:
            initial_statevector = Statevector(qc)
            print("Initial statevector:", statevector_to_str(np.array(initial_statevector)))

        print("--- new iteration", time.time())

        print("--- constructing circuit", time.time())

        print("trying to recover intensities and sources:\n", I_prev, "\n", S_prev)
        SPCircuit = state_preparation(
            I_prev, S_prev if it >= 0 else np.zeros((M,m)),
            m,
            delta_t,
            coord_idx_map, m_max_bin,
            n_qubits,
            qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla
        )
        qc = qc.compose(
            SPCircuit,
            qreg_lattice[:] + qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:]
        )
        qc.barrier()

        # At the zeroth iteration, we only do state preparation
        if it >= 1:
            qc = qc.compose(ASCircuit, qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:])
            qc.barrier()

            qc = qc.compose(AECircuit, qreg_switch[:] + qreg_ancilla[:])
            qc.barrier()

            if with_angular_redistribution:
                qc = qc.compose(ARCircuit, qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:])
                qc.barrier()

            qc = qc.compose(PCircuit, qreg_lattice[:] + qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:])
            qc.barrier()

        if "shallow" in statevector_analysis_options:
            print("--- shallow statevector analysis", time.time())
            statevector_analysis(
                qc,
                n_qubits,
                auxiliary_qubits,
                ancilla_qubits,
                verbose=verbose
            )
        if "deep" in statevector_analysis_options:
            print("--- deep statevector analysis", time.time())
            statevector_analysis_deep(
                qc,
                m,
                N,
                idx_coord_map,
                lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
                verbose=verbose
            )

        # Perform measurements to construct lattice
        qc_meas = qc.copy()
        qc_meas.measure(range(n_qubits), creg_measure)

        print("--- transpiling and running", time.time())

        qc_transpiled = transpile(qc_meas, backend, optimization_level=3)
        # print(qc_transpiled.count_ops(), sum([opcount for opcount in qc_transpiled.count_ops().values()]))
        result = backend.run(qc_transpiled, shots=int(1E4)).result()
        counts = result.get_counts(qc_transpiled)

        print("--- wrapping up", time.time())

        counts_post = dict([(measurement, count) for measurement, count in counts.items() if measurement[:3] == "000"])
        print("Full counts:", counts)
        print("Post-selected counts:", counts_post)

        lattice_I = measurements_to_lattice(
            0,
            m,
            N,
            counts,
            idx_coord_map,
            lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
            verbose=True
        )
        if np.linalg.norm(lattice_I) > 0:
            # print(lattice_I/np.linalg.norm(lattice_I))
            print(lattice_I)
        else:
            print("intensities are empty")

        if save_lattices or it == n_it:
            lattices.append(lattice_I)

        # @TODO - we have to be careful with normalization at this point because future emission
        #         steps may add a disproportionate amplitude (and thus intensity) relative
        #         to the existing radiation, as evidenced by the point_source_1D test case
        # I_prev = lattice_I.reshape((M, m), copy=True)
        I_prev = np.copy(lattice_I).reshape((M, m))

        # lattice_S = measurements_to_lattice(
        #     1,
        #     m,
        #     N,
        #     counts,
        #     idx_coord_map,
        #     lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
        #     verbose=True
        # )
        # if np.linalg.norm(lattice_S) > 0:
        #     # print(lattice_S/np.linalg.norm(lattice_S))
        #     print(lattice_S)
        # else:
        #     print("sources are empty")

        if save_circuit and it == n_it:
            qc.draw(output="mpl", filename="outputs/qc.png")

    time_stop = time.time()
    print(f"Total time: {time_stop-time_start} s")

    return lattices

