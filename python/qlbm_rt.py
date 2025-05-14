import numpy as np
np.set_printoptions(linewidth=10000)

import matplotlib.pyplot as plt

import time

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

from statevector_to_str import statevector_to_str

from qlbm_circuits import state_preparation, absorption_scattering, absorption_scattering_scattering, absorption_emission, propagation
from qlbm_utils import compute_memory_requirements, allocate_registers, compute_binary_representations, map_indices_coordinates, construct_identity_matrices
from lbm_utils import compute_grid_parameters, compute_scheme_velocities, compute_scheme_boundaries
from analysis import measurements_to_lattice, statevector_analysis

def simulate(
    I_i, S_i,
    n, m,
    N,
    n_it=1,
    delta_t=1,
    kappa=0.0,
    sigma=0.1,
    angular_redistribution_coefficients=None,
    boundary_conditions=None,
    save_lattices=False
):
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

    qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla, creg_measure, qreg_head = allocate_registers(n_qubits, n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla)
    qreg_head = QuantumRegister(1, name="H")

    I_2, I_M, I_2M, I_4M, Z_2M = construct_identity_matrices(M)

    # Absorption-Scattering, Absorption-Emission, and Propagation circuits are constant for time-independent simulation setups
    if sigma > 0.0:
        ASCircuit = absorption_scattering_scattering(
            kappa, sigma, delta_t,
            I_2,
            n_qubits_direction,
            qreg_direction, qreg_switch, qreg_ancilla
        )
    else:
        ASCircuit = absorption_scattering(
            kappa, delta_t,
            I_2,
            n_qubits_direction,
            qreg_direction, qreg_switch, qreg_ancilla
        )

    AECircuit = absorption_emission(
        I_2M, Z_2M,
        qreg_switch,
        qreg_ancilla
    )

    PCircuit = propagation(
        n, m,
        N, M,
        idxs_dir, cs,
        idx_coord_map, coord_idx_map, m_max_bin,
        angular_redistribution_coefficients,
        boundary_idxs, boundary_conditions,
        n_qubits_direction, n_qubits_switch, n_qubits_lattice,
        qreg_lattice, qreg_direction, qreg_switch,
        verbose=True
    )

    qc_0 = QuantumCircuit(
        qreg_head, qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
        creg_measure
    )

    data_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_head[:] + qreg_lattice[:]]
    auxiliary_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:]]
    lattice_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_lattice[:]]
    direction_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_direction[:]]
    switch_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_switch[:]]
    ancilla_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_ancilla[:]]
    print(data_qubits,
        auxiliary_qubits,
        lattice_qubits,
        direction_qubits,
        switch_qubits,
        ancilla_qubits,
        sep="\n"
    )

    qc_0.x(qreg_head)
    qc_0.barrier()

    # 4. Simulation
    qc_prev = qc_0
    I_prev, S_prev = np.copy(I_i), np.copy(S_i)

    lattices = []

    # Set up AerSimulator in advance (we only need to do this once)
    if "GPU" in AerSimulator().available_devices():
        # If a GPU is available, prefer cuQuantum's cuStateVec library to accelerate simulation (if built with support)
        aer_sim = AerSimulator(
            method="automatic",
            device="GPU",
            cuStateVec_enable=True
        )
        print("Using GPU")
    else:
        aer_sim = AerSimulator(
            method="automatic",
            max_parallel_threads=16,
            max_parallel_shots=1,
            max_parallel_experiments=1
        )
        print("Using CPU")

    for it in range(n_it+1):
        print(f"===== Iteration {it} =====")

        qc = qc_0
        # qc = qc_prev

        # @TODO - currently skipping this step because it takes a long time
        # initial_statevector = Statevector(qc)
        # print("Initial statevector:", statevector_to_str(np.array(initial_statevector)))

        print("--- new iteration", time.time())

        print("--- constructing circuit", time.time())

        # @TODO - figure out whether or not some sort of condition is needed here
        #         (seems like we may be getting the backwards power law problem in some tests)
        if it >= 0:
            print("trying to recover intensities and sources:\n", I_prev, "\n", S_prev)
            SPCircuit = state_preparation(
                I_prev, S_prev if it >= 0 else np.zeros((M,m)),
                m,
                delta_t,
                coord_idx_map, m_max_bin,
                n_qubits,
                qreg_head, qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla
            )
            qc = qc.compose(
                SPCircuit,
                qreg_head[:] + qreg_lattice[:] + qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:]
            )
            qc.barrier()

        recovered_statevector = np.array(Statevector(qc))
        recovered_string = statevector_to_str(recovered_statevector)
        print("Recovered statevector:", recovered_string)
        recovered_amplitudes = np.nan_to_num(recovered_statevector, nan=0.0)
        recovered_amplitudes /= np.min(recovered_amplitudes)
        recovered_outcomes = [ket[5:-1] for ket in recovered_string.split(" + ")]

        # recovered_probabilities = [np.linalg.norm(recovered_amplitude)**2 for recovered_amplitude in recovered_amplitudes]
        recovered_probabilities = [float(np.linalg.norm(float(ket[:4]))**2) for ket in recovered_string.split(" + ")]

        recovered_counts_list = [recovered_probability*1E4 for recovered_probability in recovered_probabilities]

        recovered_outcomes = [ket[5:-1] for ket in recovered_string.split(" + ")]
        recovered_counts = dict(zip(recovered_outcomes, recovered_counts_list))
        print(recovered_counts)
        print(measurements_to_lattice(
            m, N,
            recovered_counts,
            idx_coord_map,
            lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
            verbose=True
        ))

        # At the zeroth iteration, we only do state preparation
        if it >= 1:
            qc = qc.compose(ASCircuit, qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:])
            qc.barrier()
            qc = qc.compose(AECircuit, qreg_switch[:] + qreg_ancilla[:])
            qc.barrier()
            qc = qc.compose(PCircuit, qreg_lattice[:] + qreg_direction[:] + qreg_switch[:])
            qc.barrier()

        # @TODO - figure out a way to speed up this step (or not, since we don't use the output)
        # # Perform basic statevector analysis
        # print("statevector analysis", time.time())
        # sv, sv_proj, sv_lattice = statevector_analysis(qc, n_qubits, auxiliary_qubits, ancilla_qubits)

        # Perform measurements to construct lattice
        qc_meas = qc.copy()
        qc_meas.measure(range(n_qubits), creg_measure)

        print("--- transpiling and running", time.time())

        qc_transpiled = transpile(qc_meas, aer_sim, optimization_level=0)
        # print(qc_transpiled.count_ops(), sum([opcount for opcount in qc_transpiled.count_ops().values()]))
        result = aer_sim.run(qc_transpiled, shots=1E4).result()
        counts = result.get_counts(qc_transpiled)

        print("--- wrapping up", time.time())

        counts_post = dict([(measurement, count) for measurement, count in counts.items() if measurement[:3] == "000"])
        print("Full counts:", counts)
        print("Post-selected counts:", counts_post)

        lattice = measurements_to_lattice(
            m, N,
            counts,
            idx_coord_map,
            lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
            verbose=True
        )
        if np.linalg.norm(lattice) > 0:
            print(lattice/np.linalg.norm(lattice))
        else:
            print("lattice is empty")

        if save_lattices or it == n_it:
            lattices.append(lattice)

        # @TODO - we have to be careful with normalization at this point because guarantee future
        #         emission steps may add a disproportionate amplitude (and thus intensity) relative
        #         to the existing radiation, as evidenced by the point_source_1D test case
        I_prev = lattice.reshape((M, m), copy=True)
        norm = np.linalg.norm(I_prev)
        print(norm)
        if norm > 0:
            I_prev /= norm

    return lattices

