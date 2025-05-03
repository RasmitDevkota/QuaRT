import numpy as np
np.set_printoptions(linewidth=1000)

import matplotlib.pyplot as plt

from qiskit.quantum_info import Statevector, partial_trace

from statevector_to_str import statevector_to_str

def measurements_to_lattice(
	m,
	N,
	counts,
	idx_coord_map,
	lattice_qubits, direction_qubits, switch_qubits, ancilla_qubits,
	verbose=False
):
	lattice = np.zeros((*N, m))

	for outcome, count in counts.items():
		outcome_ancilla_bin = "".join(outcome[l] for l in ancilla_qubits)
		outcome_switch_bin = "".join(outcome[l] for l in switch_qubits)
		if outcome[:3] == "000" and outcome[switch_qubits[0]] == "0":
			lattice_point_bin = "".join(outcome[l] for l in lattice_qubits)
			lattice_point = idx_coord_map[lattice_point_bin]

			lattice_direction_bin = "".join(outcome[l] for l in direction_qubits)
			mu = int(lattice_direction_bin, 2)
			
			lattice[*lattice_point, mu] += count
			
			if verbose:
				print(f"{outcome} -> {lattice_point_bin} ({lattice_point}), {mu}: {count}")

	return lattice

def statevector_analysis(qc, n_qubits, auxiliary_qubits, ancilla_qubits, verbose=False):
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

