import numpy as np
from qsim.state_vector import StateVector
from qsim.gates import X

class QuantumCircuit:
    def __init__(self, num_qubits=None, state_vector=None, rng=None):
        if num_qubits is None and state_vector is None:
            raise ValueError("Either num_qubits or state_vector must be provided.")
        
        if num_qubits is not None and state_vector is not None:
            raise ValueError("Provide either num_qubits or state_vector, not both.")
        
        self.state_vector = StateVector(state_vector.num_qubits, state_vector.state, state_vector.rng) if state_vector is not None else StateVector(num_qubits, rng=rng)
        self.num_qubits = self.state_vector.num_qubits
        self.gates = []

    def add_gate(self, gate, target, gate_name=None, controls=None):
        if controls is None:
            controls_list = []
        elif isinstance(controls, (int, np.integer)):
            controls_list = [controls]
        else:
            controls_list = list(controls)

        if target < 0 or target >= self.num_qubits:
            raise ValueError("Target qubit index out of range.")

        
        for control in controls_list:
            if not isinstance(control, (int, np.integer)):
                raise ValueError("All controls must be non-negative integers")
            if control < 0 or control >= self.num_qubits:
                raise ValueError("At least one of the controls is out of range")
            if control == target:
                raise ValueError("Control and target qubits must be different.")
        if len(controls_list) != len(set(controls_list)):
            raise ValueError("All controls must be unique")

        self.gates.append((gate, target, gate_name, controls_list))

    def add_swap_gate(self, qubit1, qubit2):
        if qubit1 < 0 or qubit1 >= self.num_qubits:
            raise ValueError("Qubit1 index out of range.")
        
        if qubit2 < 0 or qubit2 >= self.num_qubits:
            raise ValueError("Qubit2 index out of range.")
        
        if qubit1 == qubit2:
            raise ValueError("Qubit1 and Qubit2 must be different.")

        self.add_gate(X, qubit1, gate_name="SWAP(1/3)", controls=qubit2)
        self.add_gate(X, qubit2, gate_name="SWAP(2/3)", controls=qubit1)
        self.add_gate(X, qubit1, gate_name="SWAP(3/3)", controls=qubit2)

    def apply_circuit(self):
        sv = StateVector(self.num_qubits, self.state_vector.state, self.state_vector.rng)
        for gate, target, _, controls in self.gates:
            if controls == []:
                sv.apply_gate(gate, target)
            else:
                sv.apply_multi_controlled_gate(gate, controls, target)
        return sv

    def sample(self, shots):

        if not isinstance(shots, int) or shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")
        
        sv = self.apply_circuit()
        probabilities = sv.probabilities()
        indices = sv.rng.choice(len(probabilities), size=shots, p=probabilities)
        counts = {}
        for index in indices:
            bitstring = format(index, f'0{self.num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def __str__(self):
        first_line = f"Quantum Circuit with {self.num_qubits} qubits and {len(self.gates)} gates."
        gate_lines = []
        for _, target, gate_name, controls in self.gates:
            if controls == []:
                gate_lines.append(f"{gate_name or 'Unnamed Gate'} on qubit {target}")
            elif len(controls) == 1:
                gate_lines.append(f"{gate_name or 'Unnamed Controlled Gate'} on qubit {target} controlled by qubit {controls[0]}")
            else:
                controls_text = ", ".join(str(control) for control in controls)
                gate_lines.append(f"{gate_name or 'Unnamed Multi Controlled Gate'} on qubit {target} controlled by qubits {controls_text}")
        return "\n".join([first_line] + gate_lines)