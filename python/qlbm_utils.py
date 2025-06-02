import numpy as np

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit

from lbm_utils import compute_grid_parameters

def compute_memory_requirements(m, M_0, verbose=False):
	n_qubits_lattice = int(np.ceil(np.log2(M_0)))
	n_qubits_boundary = 1
	n_qubits_direction = int(np.ceil(np.log2(m)))
	n_qubits_switch = 1
	n_qubits_ancilla = 1 + 2 + 2 + 1 + 1 # 1 - AS, 2 - AE, 2 - AR, 1 - BC, 1 - @TODO unknown?
	n_qubits = n_qubits_lattice + n_qubits_boundary + n_qubits_direction + n_qubits_switch + n_qubits_ancilla

	if verbose:
		print(f"Total qubits: {n_qubits}")
		print(f"Lattice qubits: {n_qubits_lattice}")
		print(f"Boundary qubits: {n_qubits_boundary}")
		print(f"Direction qubits: {n_qubits_direction}")
		print(f"Switch qubits: {n_qubits_switch}")
		print(f"Ancilla qubits: {n_qubits_ancilla}")

	return n_qubits, n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla

def allocate_registers(
    n_qubits,
    n_qubits_lattice, n_qubits_boundary, n_qubits_direction, n_qubits_switch, n_qubits_ancilla
):
	qreg_lattice = QuantumRegister(n_qubits_lattice, name="L")
	qreg_boundary = QuantumRegister(n_qubits_boundary, name="B")
	qreg_direction = QuantumRegister(n_qubits_direction, name="D")
	qreg_switch = QuantumRegister(n_qubits_switch, name="S")
	qreg_ancilla = QuantumRegister(n_qubits_ancilla, name="A")
	creg_measure = ClassicalRegister(n_qubits, name="C")

	return qreg_lattice, qreg_boundary, qreg_direction, qreg_switch, qreg_ancilla, creg_measure

def compute_binary_representations(m, M, verbose=False):
	coordinate_max = M
	coordinate_max_bin = bin(coordinate_max-1)[2:]

	m_max_bin = bin(m-1)[2:]

	if verbose:
		print("Maximum coordinate binary representation:", int(coordinate_max_bin, 2), len(coordinate_max_bin))
		print("Maximum direction binary representation:", int(m_max_bin, 2), len(m_max_bin))

	return coordinate_max_bin, m_max_bin

def map_indices_coordinates(N, coordinate_max_bin, verbose=False):
	# Construct mappings from binary coordinate indices to coordinates
	idx_coord_map = {}

	c = 0
	for coord in np.ndindex(tuple(N)):
		idx_coord_map[bin(c)[2:].zfill(len(coordinate_max_bin))] = coord
		c += 1

	# Construct mappings from binary coordinate indices to coordinates
	coord_idx_map = dict(zip(idx_coord_map.values(), idx_coord_map.keys()))

	if verbose:
		print("Index->coordinate map:", idx_coord_map)
		print("---")
		print("Coordinate->index map:", coord_idx_map)

	return idx_coord_map, coord_idx_map

def construct_identity_matrices(M):
	I_2 = np.eye((2))
	I_M = np.eye((M))
	I_2M = np.kron(I_2, I_M)
	I_4M = np.kron(I_2, I_2M)
	Z_2M = np.zeros_like(I_2M)

	return I_2, I_M, I_2M, I_4M, Z_2M

def vector_to_lattice(v):
	raise NotImplementedError()

def lattice_to_vector(lattice):
	N = lattice.shape[:-1]
	n = len(N)
	m = lattice.shape[-1]

	M_0, M = compute_grid_parameters(n, N)
	coordinate_max_bin, m_max_bin = compute_binary_representations(m, M)
	idx_coord_map, coord_idx_map = map_indices_coordinates(N, coordinate_max_bin)

	v = np.zeros((M, m))

	for coord in np.ndindex(N):
		idx_bin = coord_idx_map[coord]
		idx_dec = int(f"0b{idx_bin}", 2)
		v[idx_dec] = lattice[coord]

	return v

