import numpy as np

class StateVector:
    def __init__(self, num_qubits, state=None, rng=None):
        state = np.array(state, dtype=complex) if state is not None else None
        
        if state is not None and len(state) != 2 ** num_qubits:
            raise ValueError("State vector length must be 2^num_qubits.")
        
        if state is not None and not np.isclose(np.sum(np.abs(state) ** 2), 1):
            raise ValueError("State vector must be normalized.")

        if isinstance(num_qubits, int) and num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")

        self.num_qubits = num_qubits
        
        if state is not None:
            self.state = state
        else:
            self.state = np.zeros(2 ** num_qubits, dtype=complex)
            self.state[0] = 1 # Initialize to |0,0...0>
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply_gate(self, gate, target):
        if target < 0 or target >= self.num_qubits:
            raise ValueError("Target qubit index out of range.")
        
        distance = 2 ** target
        size = len(self.state)
        block_size = distance * 2
        for block in range(0, size, block_size):
            for i in range(distance):
                first_cell = block + i
                second_cell = first_cell + distance
                temp_vector = np.array([self.state[first_cell], self.state[second_cell]], dtype=complex)
                new_vector = gate @ temp_vector
                self.state[first_cell] = new_vector[0]
                self.state[second_cell] = new_vector[1]

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self):
        probabilities = self.probabilities()
        index = self.rng.choice(len(self.state), p=probabilities)
        self.state = np.zeros_like(self.state)
        self.state[index] = 1
        return index

    def measure_single(self, target):
        if target < 0 or target >= self.num_qubits:
            raise ValueError("Target qubit index out of range.")
        
        probabilities = self.probabilities()
        size = len(self.state)
        distance = 2 ** target
        block_size = distance * 2
        prob_0 = 0.0
        for block in range(0, size, block_size):
            for i in range(distance):
                first_cell = block + i
                prob_0 += probabilities[first_cell]
        r = self.rng.random()
        for block in range(0, size, block_size):
            for i in range(distance):
                first_cell = block + i
                second_cell = first_cell + distance
                if r < prob_0:
                    self.state[second_cell] = 0
                else:
                    self.state[first_cell] = 0
        self.state /= np.linalg.norm(self.state)
        return 0 if r < prob_0 else 1