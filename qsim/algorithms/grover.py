import numpy as np
from qsim.gates import Z, H, X
from qsim.QuantumCircuit import QuantumCircuit
from qsim.state_vector import StateVector
from qsim.algorithms.oracles import *

def diffuser(sv):
    circuit = QuantumCircuit(state_vector=sv)
    qubits_list = []
    for qubit in range(circuit.num_qubits):
        qubits_list.append(qubit)
    circuit.add_gate(H, qubits_list, "H")
    circuit.add_gate(X, qubits_list, "X")
    circuit.add_gate(Z, qubits_list[-1], "Multi Controlled Z", qubits_list[:-1])
    circuit.add_gate(X, qubits_list, "X")
    circuit.add_gate(H, qubits_list, "H")
    return circuit.apply_circuit()

def grover(f, num_qubits, k=None):
    if (not isinstance(num_qubits, (int, np.integer))) or num_qubits <= 0:
        raise ValueError("Number of qubits must be a positive integer")
    if k is not None:
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("Number of repetitons must be a non-negative integer")
    
    sv = StateVector(num_qubits)
    oracle_matrix = make_oracle_matrix(f, num_qubits)
    m = 0
    n = 2 ** num_qubits
    for q in range(num_qubits):
        sv.apply_gate(H, q)
    for i in range(n):
        if f(i):
            m += 1
    if m == 0:
        raise ValueError("There are no solution for the provided function in the range of provided qubits number")
    if k is None:
        theta = np.arcsin(np.sqrt(m / n))
        k = np.pi / (4 * theta)
    for _ in range(int(np.floor(k))):
        sv.apply_full_matrix(oracle_matrix)
        sv = diffuser(sv)
    return sv
    

    