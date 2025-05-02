import numpy as np

n = 2
m = 8
N = [8, 8]
M = np.prod(N)

n_qubits_lattice = int(np.ceil(np.log2(M_0)))
n_qubits_direction = int(np.ceil(np.log2(m)))
n_qubits_switch = 1
n_qubits_ancilla = 4
n_qubits = 1 + n_qubits_lattice + n_qubits_direction + n_qubits_switch + n_qubits_ancilla # counting head qubit

I_i = np.zeros(shape=(M, m))
S_i = np.zeros(shape=(M, m))

initial_statevector = np.zeros((2**n_qubits))

anc_bin = "000"

for c, (coordinate_idx, coordinate_bin) in enumerate(coord_idx_map.items()):
    coordinate_bin = coordinate_bin[::-1]

    for mu in range(m):
        mu_bin = bin(mu)[2:].zfill(len(m_max_bin))#[::-1]

        for s_bin in range(2):
            if s_bin == 0:
                prob_amp = I_i[c, mu]
            else:
                prob_amp = 0.5 * delta_t * S_i[c, mu]

            idx_bin = f"0b{anc_bin}{s_bin}{mu_bin}{coordinate_bin}0"
            idx_dec = int(idx_bin, 2)
            initial_statevector[idx_dec] = prob_amp

norm = np.linalg.norm(initial_statevector)
print(norm)
if norm > 0:
    initial_statevector /= norm
