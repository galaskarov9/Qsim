import numpy as np
from qsim.state_vector import StateVector

class QuantumCircuit:
    def __init__(self, num_qubits=None, state_vector=None, rng=None):
        if num_qubits is None and state_vector is None:
            raise ValueError("Either num_qubits or state_vector must be provided.")
        
        if num_qubits is not None and state_vector is not None:
            raise ValueError("Provide either num_qubits or state_vector, not both.")
        
        self.state_vector = StateVector(state_vector.num_qubits, state_vector.state, state_vector.rng) if state_vector is not None else StateVector(num_qubits, rng=rng)
        self.num_qubits = self.state_vector.num_qubits
        self.gates = []

    def add_gate(self, gate, target, gate_name=None):
        if target < 0 or target >= self.num_qubits:
            raise ValueError("Target qubit index out of range.")
        
        self.gates.append((gate, target, gate_name))

    def apply_gates(self):
        sv = StateVector(self.num_qubits, self.state_vector.state, self.state_vector.rng)
        for gate, target, _ in self.gates:
            sv.apply_gate(gate, target)
        return sv

    def __str__(self):
        first_line = f"Quantum Circuit with {self.num_qubits} qubits and {len(self.gates)} gates."
        gate_lines = [f"  - {gate_name or 'Unnamed Gate'} on qubit {target}" for _, target, gate_name in self.gates]
        return "\n".join([first_line] + gate_lines)