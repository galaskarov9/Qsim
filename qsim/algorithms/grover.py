import numpy as np
from qsim.gates import Z, H, X
from qsim.QuantumCircuit import QuantumCircuit

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

    