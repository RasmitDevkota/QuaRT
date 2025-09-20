import numpy as np
np.set_printoptions(linewidth=1000)

import matplotlib.pyplot as plt

import re

from qiskit.quantum_info import Statevector, partial_trace

from statevector_to_str import statevector_to_str

def measurements_to_lattice(
	quantity_idx,
	m,
    N,
	counts, shots,
	idx_coord_map,
	lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
	verbose=False
):
	lattice = np.zeros((*N, m))

	for outcome, count in counts.items():
		outcome_ancilla_bin = "".join(outcome[l] for l in ancilla_qubits)
		outcome_switch_bin = "".join(outcome[l] for l in switch_qubits)

		if "1" not in outcome_ancilla_bin and outcome[switch_qubits[0]] == str(quantity_idx):
			lattice_point_bin = "".join(outcome[l] for l in lattice_qubits)
			lattice_point = idx_coord_map[lattice_point_bin]

			lattice_direction_bin = "".join(outcome[l] for l in direction_qubits)#[::-1]
			mu = int(lattice_direction_bin, 2)
			
			lattice[*lattice_point, mu] += np.sqrt(count/shots)
			
			if verbose:
				print(f"Counts for outcome {outcome} at point {lattice_point_bin} ({lattice_point}), in direction {mu} ({lattice_direction_bin}): {count}")

	return lattice

def statevector_analysis(
    qc,
    n_qubits,
    auxiliary_qubits,
    ancilla_qubits,
    verbose=False
):
	print("Performing statevector analysis...")

	sv = Statevector(qc)

	print("Constructed Statevector from QuantumCircuit")

	if verbose:
		print(np.count_nonzero(sv), np.size(sv))
		print(sv.dim, sv.num_qubits)
		print(sv.purity())

	print("Full statevector:", statevector_to_str(np.array(sv)))

	sv_subspace = Statevector([1,0]).tensor(Statevector([1,0])).tensor(Statevector([1,0]))
	P_subspace = sv_subspace.to_operator()

	sv_proj = sv.evolve(P_subspace, [n_qubits - q - 1 for q in ancilla_qubits])

	if verbose:
		print(np.count_nonzero(sv_proj), np.size(sv_proj))
		print(sv_proj.dim, sv_proj.num_qubits)
		print(sv_proj.purity())

	print("Projected statevector:", statevector_to_str(np.array(sv_proj)))

	sv_lattice = partial_trace(sv_proj, auxiliary_qubits)

	if verbose:
		print(np.count_nonzero(sv_lattice), np.size(sv_L))
		print(sv_lattice.dim, sv_lattice.num_qubits)
		print(sv_lattice.purity())

	# print("Lattice data statevector:", statevector_to_str(np.array(sv_lattice)))
	# print("Lattice data statevector:", np.array(sv_lattice))

	return sv, sv_proj, sv_lattice

def statevector_analysis_deep(
    qc,
    m,
    N,
    idx_coord_map,
    lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
    verbose=False
):
    resultant_statevector = np.array(Statevector(qc))
    resultant_string = statevector_to_str(resultant_statevector)
    print("Statevector:", resultant_string)

    resultant_pairs = re.findall(r"([\-0-9\.]*)(\|[01]+〉)", resultant_string)
    resultant_amplitudes = []
    resultant_outcomes = []
    for resultant_amplitude, resultant_outcome in resultant_pairs:
        resultant_amplitudes.append(float(resultant_amplitude) if len(resultant_amplitude) > 0 else 1.0)
        resultant_outcomes.append(resultant_outcome[1:-1])

    resultant_probabilities = [float(np.linalg.norm(resultant_amplitude)**2) for resultant_amplitude in resultant_amplitudes]
    resultant_counts_list = [resultant_probability*1E4 for resultant_probability in resultant_probabilities]

    resultant_counts = dict(zip(resultant_outcomes, resultant_counts_list))
    print("Counts dictionary:", resultant_counts)

    print("Intensity lattice:\n", measurements_to_lattice(
        0,
        m, N,
        resultant_counts,
        idx_coord_map,
        lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
        verbose=True
    ))

    print("Source lattice:\n", measurements_to_lattice(
        1,
        m, N,
        resultant_counts,
        idx_coord_map,
        lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
        verbose=True
    ))

