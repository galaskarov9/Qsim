import numpy as np
from qsim.gates import H, P

def qft(circuit, qubits_list):
    n = len(qubits_list)
    if n != len(set(qubits_list)):
        raise ValueError("All qubits in the list must be unique")

    for m in range(n-1, -1, -1):
        circuit.add_gate(H, qubits_list[m], "H")
        for k in range(m -1, -1, -1):
            theta = np.pi / (2 ** (m - k))
            circuit.add_gate(P(theta), qubits_list[m], "controlled phase shift", qubits_list[k])
    for i in range(n // 2):
        q_i = qubits_list[i]
        q_j = qubits_list[n - 1 - i]
        circuit.add_swap_gate(q_i, q_j)