import numpy as np

class Qubit:
    def __init__(self, state=None, rng=None):
        self.state = state if state is not None else np.array([1, 0], dtype=complex)
        self.rng = rng if rng is not None else np.random.default_rng()

    def apply_gate(self, gate):
        self.state = gate @ self.state

    def probabilities(self):
        return np.abs(self.state) ** 2

    def measure(self):
        r = self.rng.random()
        if r < self.probabilities()[0]:
            self.state = np.array([1, 0], dtype=complex)  # Collapse to |0>
            return 0
        else:
            self.state = np.array([0, 1], dtype=complex)  # Collapse to |1>
            return 1