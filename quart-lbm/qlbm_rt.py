import time

import numpy as np
np.set_printoptions(threshold=np.inf)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qlbm_circuits import state_preparation, absorption_scattering, absorption_emission, angular_redistribution, special_boundary_conditions, propagation
from qlbm_utils import compute_memory_requirements, allocate_registers, compute_binary_representations, map_indices_coordinates, construct_identity_matrices
from lbm_utils import compute_grid_parameters, compute_scheme_velocities, compute_scheme_adjacencies, compute_scheme_boundaries
from analysis import measurements_to_lattice, statevector_analysis, statevector_analysis_deep

from typing import TYPE_CHECKING
from typing import Iterable

def simulate(
    I_i, S_i,
    n, m,
    N,
    n_timesteps: int = 1,
    delta_t: float = 1.0,
    kappa: float = 0.0,
    sigma: float = 0.0,
    angular_redistribution_coefficients: Iterable[float] | None = None,
    boundary_conditions: Iterable | None = None,
    save_lattices: bool = False,
    save_circuit: bool = False,
    save_name: str | None = None,
    statevector_analysis_options: Iterable | None = None,
    verbose: bool = False
) -> tuple[Iterable, Iterable, Iterable]:
    """Main simulation function for lattice Boltzmann radiative transfer.

    Args:
        I_i: Lattice of initial intensities.
        S_i: Lattice of initial sources>
        n: Number of simulation dimensions.
        m: Number of direction vectors for the lattice Boltzmann method.
        N: Grid dimensions.
        n_timesteps: Number of simulation timesteps (does not include timestep 0, which can exclude certain processes).
        delta_t: Temporal resolution of the simulation.
        kappa: Constant or array of absorption coefficients.
        sigma: Constant or array of scattering coefficients.
        angular_redistribution_coefficents: Angular redistribution coefficients for use if AR is to be applied.
        boundary_conditions: Boundary condition specifications.
        save_lattices: Specifies whether to save the lattice at intermediate timesteps.
        save_circuit: Specifies whether to save an image of the circuit at each timestep.
        save_name: Unique name to be used in save files.
        statevector_analysis_options: Any options to be used during statevector simulation.
        verbose: Specifies whether or not to print verbose information.

    Returns:
        Intensity lattices, source lattices, and statevector norms
    """

    # Record start time
    time_start = time.time()

    # 0. Input parsing
    if boundary_conditions is None:
        boundary_conditions = [("periodic", None)]*4

    if statevector_analysis_options is None:
        statevector_analysis_options = []

    # 1. Lattice utilities
    M_0, M = compute_grid_parameters(n, N)
    idxs_dir, cs = compute_scheme_velocities(n, m)
    adjacencies = compute_scheme_adjacencies(n, m, idxs_dir, cs)
    boundary_idxs = compute_scheme_boundaries(n, m, idxs_dir, cs)
    coordinate_max_bin, m_max_bin = compute_binary_representations(m, M)
    idx_coord_map, coord_idx_map = map_indices_coordinates(N, coordinate_max_bin)

    # 2. Initial conditions, sources, boundaries
    if I_i is None:
        I_i = np.zeros(shape=(M, m))

    if S_i is None:
        S_i = np.zeros(shape=(M, m))

    # 3. Quantum system construction
    # Only perform AS if sigma (the scattering coefficient) is not close to 0.0
    perform_AS = not np.isclose(sigma, 0.0)

    # Only perform AS if the angular redistribution_coefficients are specified and not all close to 1.0
    perform_AR = angular_redistribution_coefficients is not None and not np.isclose(angular_redistribution_coefficients[0], 1.0)

    if perform_AR:
        norm_factor = m/2
    else:
        norm_factor = m

    # Only perform (special) BCs if at least one boundary condition is specified to not be periodic
    perform_BC = (
        (boundary_conditions is not None and any([boundary_condition[0] != "periodic" for boundary_condition in boundary_conditions])) or
        (isinstance(kappa, float) and kappa > 0.0) or
        (hasattr(kappa, "__iter__") and np.count_nonzero(kappa) > 0)
    )

    n_qubits, n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla, ancilla_idxs_AS, ancilla_idxs_AE, ancilla_idxs_AR, ancilla_idxs_BC = compute_memory_requirements(
        m, M_0,
        include_AS=perform_AS,
        include_AR=perform_AR,
        include_BC=perform_BC,
    )

    qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla, creg_measure = allocate_registers(n_qubits, n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla)

    I_2, I_M, I_2M, I_4M, Z_2M = construct_identity_matrices(M)

    # Absorption-Scattering, Absorption-Emission, Angular Redistribution, Propagation, and Boundary Condition circuits are constant for time-independent simulation setups
    ASCircuit = None
    if (
        (isinstance(sigma, float) and sigma > 0.0) or
        (hasattr(sigma, "__iter__") and np.count_nonzero(sigma) > 0)
        and perform_AS
    ):
        print("Performing absorption+scattering")

        ASCircuit = absorption_scattering(
            kappa, sigma,
            delta_t,
            I_2,
            n_qubits_direction,
            qreg_direction, qreg_switch, qreg_ancilla,
            ancilla_idxs_AS,
        )

    AECircuit = absorption_emission(
        I_2M, Z_2M,
        qreg_switch, qreg_ancilla,
        ancilla_idxs_AE,
    )

    PCircuit_list, PQubit_list = propagation(
        n, m,
        N, M,
        idxs_dir, cs,
        idx_coord_map, coord_idx_map, m_max_bin,
        n_qubits_lattice, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
        qreg_lattice, qreg_direction, qreg_switch, qreg_ancilla,
        verbose=False
    )

    ARCircuit = None
    if perform_AR:
        print("Performing angular redistribution")

        # @TEST
        ARCircuit = angular_redistribution(
        # ARCircuit = angular_redistribution_3d(
            m,
            delta_t,
            angular_redistribution_coefficients,
            idxs_dir, cs,
            adjacencies,
            idx_coord_map, coord_idx_map, m_max_bin,
            n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
            qreg_direction, qreg_switch, qreg_ancilla,
            ancilla_idxs_AR,
        )

    BCCircuit = None
    if perform_BC:
        print(f"Performing special boundary conditions: {boundary_conditions}")

        BCCircuit = special_boundary_conditions(
            n, m,
            N, M,
            cs,
            idx_coord_map, coord_idx_map, m_max_bin,
            boundary_idxs,
            n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla,
            qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
            ancilla_idxs_BC,
            verbose=False
        )

    qc_0 = QuantumCircuit(
        qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
        creg_measure
    )

    # Log qubit usage
    lattice_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_lattice[:]]
    auxiliary_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_boundary[:] + qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:]]
    direction_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_direction[:]]
    boundary_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_boundary[:]]
    switch_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_switch[:]]
    ancilla_qubits = [n_qubits-qc_0.qubits.index(qubit)-1 for qubit in qreg_ancilla[:]]
    print(
        f"Lattice qubits: {lattice_qubits}",
        f"Auxiliary qubits: {auxiliary_qubits}",
        f"Boundary qubits: {boundary_qubits}",
        f"Direction qubits: {direction_qubits}",
        f"Switch qubits: {switch_qubits}",
        f"Ancilla qubits: {ancilla_qubits}",
        sep="\n"
    )

    # 4. Simulation
    I_prev, S_prev = np.copy(I_i), np.copy(S_i)

    lattices_I = []
    lattices_S = []
    norms = []

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
        # @TODO - pick optimal parameters
        # @TODO - should be exposed as user options
        backend = AerSimulator(
            method="automatic",
            # method="matrix_product_state",
            max_parallel_threads=0,
            max_parallel_experiments=0,
            max_parallel_shots=1
        )
        print("Using CPU")

    # Simulate
    for timestep in range(n_timesteps+1):
        print(f"========== Timestep {timestep} / Time {timestep*delta_t} ==========")

        print("--- Constructing circuit", time.time())

        qc = qc_0.copy()

        global_norm = 1

        # print("trying to recover intensities and sources:\n", I_prev, "\n", S_prev)
        print(f"State prep using norm_factor={norm_factor}")
        SPCircuit, norm = state_preparation(
            I_prev, S_prev if timestep >= 0 else np.zeros((M,m)),
            n, m,
            N,
            kappa,
            delta_t,
            coord_idx_map, m_max_bin,
            boundary_idxs, boundary_conditions,
            n_qubits, n_qubits_ancilla,
            qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla,
            verbose=False,
            norm_factor=norm_factor
        )
        print("Composing SP")
        qc.compose(SPCircuit, qreg_lattice[:] + qreg_boundary[:] + qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:], inplace=True)
        qc.barrier()

        norms.append(norm)
        global_norm *= norm

        if "prepared" in statevector_analysis_options:
            print("--- Initial statevector analysis", time.time())
            statevector_analysis_deep(
                qc,
                m,
                N,
                idx_coord_map,
                lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
                verbose=True
            )

        # @TODO - at the zeroth timestep, should we only perform state preparation?
        if timestep >= 0:
            if ASCircuit is not None:
                print("Composing AS", time.time())
                qc.compose(ASCircuit, qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:], inplace=True)
                qc.barrier()

            print("Composing AE", time.time())
            qc.compose(AECircuit, qreg_switch[:] + qreg_ancilla[:], inplace=True)
            qc.barrier()

            # AE adds a factor of 1/2
            global_norm *= 2

            print("Composing P", time.time())
            for PCircuit, PQubits in zip(PCircuit_list, PQubit_list):
                qc.compose(PCircuit, PQubits, inplace=True)
            qc.barrier()

            if ARCircuit is not None:
                print("Composing AR", time.time())
                qc.compose(ARCircuit, qreg_direction[:] + qreg_switch[:] + qreg_ancilla[:], inplace=True)
                qc.barrier()

            if BCCircuit is not None:
                print("Composing BC", time.time())
                qc.compose(BCCircuit, qreg_boundary[:] + qreg_switch[:] + qreg_ancilla[:], inplace=True)
                qc.barrier()

        if "shallow" in statevector_analysis_options:
            print("--- Shallow statevector analysis", time.time())
            statevector_analysis(
                qc,
                n_qubits,
                auxiliary_qubits,
                ancilla_qubits,
                verbose=verbose
            )
        if "deep" in statevector_analysis_options:
            print("--- Deep statevector analysis", time.time())
            statevector_analysis_deep(
                qc,
                m,
                N,
                idx_coord_map,
                lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
                verbose=True
            )

        # Perform measurements to construct lattice
        qc_meas = qc
        qc_meas.measure(range(n_qubits), creg_measure)

        if save_circuit:
            qc_meas.draw(output="mpl", filename="outputs/qc.png")

        print("--- Transpiling", time.time())

        shots = 1E6
        qc_transpiled = transpile(qc_meas, backend, optimization_level=0)
        if verbose:
            print(qc_transpiled.count_ops(), sum([opcount for opcount in qc_transpiled.count_ops().values()]))

        print("--- Running", time.time())
        result = backend.run(qc_transpiled, shots=shots).result()
        print(result)

        print("--- Processing results", time.time())

        counts = result.get_counts(qc_transpiled)
        counts_post = dict([(measurement, count) for measurement, count in counts.items() if measurement.startswith("0"*n_qubits_ancilla)])
        shots_post = sum(list(counts_post.values()))
        # print("Full counts:", counts)
        print("Post-selected counts:", counts_post)
        print("Total post-selected counts:", shots_post)

        lattice_I = measurements_to_lattice(
            0,
            m,
            N,
            counts, shots,
            idx_coord_map,
            lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
            verbose=True
        )
        lattice_I *= global_norm
        if verbose:
            if np.linalg.norm(lattice_I) > 0:
                print(lattice_I)
            else:
                print("Intensities are empty")

        I_prev = np.copy(lattice_I).reshape((M, m))

        lattice_S = measurements_to_lattice(
            1,
            m,
            N,
            counts, shots,
            idx_coord_map,
            lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
            verbose=True
        )
        # @TODO - check
        lattice_S *= m / delta_t * global_norm
        if verbose:
            if np.linalg.norm(lattice_S) > 0:
                print(lattice_S)
            else:
                print("Sources are empty")

        S_prev = np.copy(lattice_S).reshape((M, m))

        # Save output as specified
        if save_lattices or timestep == n_timesteps:
            lattices_I.append(lattice_I)
            lattices_S.append(lattice_S)

            if n == 1:
                filename = f"outputs-intermediate/lattice_{save_name}_{timestep}-{n_timesteps}_{N[0]}_{m}.npy"
            else:
                N_str = "-".join([str(N_i) for N_i in N])
                filename = f"outputs-intermediate/lattice_{save_name}_{timestep}-{n_timesteps}_{N_str}_{m}.npy"

            np.save(filename, lattices_I)

            print(f"Saved intermediate data to {filename}")

    time_stop = time.time()
    print(f"Total time: {time_stop-time_start} s")

    return lattices_I, lattices_S, norms

