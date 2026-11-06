import numpy as np
from qsim.qubit import Qubit

X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

def test_qubit_initialization():
    qubit = Qubit()
    assert np.allclose(qubit.state, np.array([1, 0], dtype=complex))

def test_qubit_apply_gate():
    qubit1 = Qubit()
    qubit2 = Qubit()
    qubit1.apply_gate(X)
    qubit2.apply_gate(H)
    assert np.allclose(qubit1.state, np.array([0, 1], dtype=complex))
    assert np.allclose(qubit2.state, np.array([1, 1], dtype=complex) / np.sqrt(2))

def test_qubit_probabilities():
    qubit = Qubit()
    qubit.apply_gate(H)
    probs = qubit.probabilities()
    assert np.allclose(probs, np.array([0.5, 0.5]))

def test_qubit_measurement_for_zero():
    qubit = Qubit()
    for _ in range(100):
        result = qubit.measure()
        assert result == 0

def test_qubit_measurement_for_one():
    qubit = Qubit()
    qubit.apply_gate(X)
    for _ in range(100):
        result = qubit.measure()
        assert result == 1

def test_qubit_measurement_randomness():
    rng = np.random.default_rng(seed=42)
    zero_count = 0
    for _ in range(1000):
        qubit =Qubit(rng=rng)
        qubit.apply_gate(H)
        result = qubit.measure()
        if result == 0:
            zero_count += 1
    assert 400 < zero_count < 600  # Expect roughly half to be 0

def test_complex_qubit_X():
    qubit = Qubit(state=np.array([0.8j, 0.6], dtype=complex))
    qubit.apply_gate(X)
    assert np.allclose(qubit.state, np.array([0.6, 0.8j], dtype=complex))