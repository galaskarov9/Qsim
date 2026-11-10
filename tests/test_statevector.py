import numpy as np
from qsim.state_vector import StateVector
from qsim.gates import X, H

def test_statevector_initialization():
    assert np.allclose(StateVector(2).state, np.array([1, 0, 0, 0], dtype=complex)), "Failed to initialize state vector for 2 qubits"
    assert np.allclose(StateVector(2, state=[1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]).state, np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)), "Failed to initialize state vector with custom state for 2 qubits"
    assert np.allclose(StateVector(3, state=[1/np.sqrt(8)]*8).state, np.array([1/np.sqrt(8)]*8, dtype=complex)), "Failed to initialize state vector with custom state for 3 qubits"

def test_apply_gate():
    sv = StateVector(2)
    sv.apply_gate(X, 0)
    expected_state = np.array([0, 1, 0, 0], dtype=complex)
    assert np.allclose(sv.state, expected_state), "Failed to apply X gate on qubit 0"

    sv = StateVector(2)
    sv.apply_gate(H, 0)
    expected_state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0], dtype=complex)
    assert np.allclose(sv.state, expected_state), "Failed to apply H gate on qubit 0"

    sv = StateVector(3)
    sv.apply_gate(H, 1)
    expected_state = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0, 0, 0, 0, 0], dtype=complex)
    assert np.allclose(sv.state, expected_state), "Failed to apply H gate on qubit 1"

    sv = StateVector(4)
    sv.apply_gate(X, 2)
    expected_state = np.zeros(16, dtype=complex)
    expected_state[4] = 1
    assert np.allclose(sv.state, expected_state), "Failed to apply X gate on qubit 2"

    sv = StateVector(5)
    sv.apply_gate(H, 4)
    expected_state = np.zeros(32, dtype=complex)
    expected_state[0] = 1/np.sqrt(2)
    expected_state[16] = 1/np.sqrt(2)
    assert np.allclose(sv.state, expected_state), "Failed to apply H gate on qubit 4"

def test_measure_basis_state():
    sv = StateVector(2, state=[0, 0, 1, 0])
    result = sv.measure()
    assert result == 2, f"expected 2, got {result}"
    assert np.allclose(sv.state, [0, 0, 1, 0]), "state should not change"


def test_measure_collapse():
    sv = StateVector(2)
    sv.apply_gate(H, 0)
    result = sv.measure()
    assert result in (0, 1), f"impossible result {result}"
    expected_state = np.zeros(4, dtype=complex)
    expected_state[result] = 1
    assert np.allclose(sv.state, expected_state), "state did not collapse to measured cell"


def test_measure_statistics():
    rng = np.random.default_rng(42)
    shots = 10000
    counts = np.zeros(4)
    for _ in range(shots):
        sv = StateVector(2, state=[0.6, 0, 0.8, 0], rng=rng)
        counts[sv.measure()] += 1
    frequencies = counts / shots
    assert np.allclose(frequencies, [0.36, 0, 0.64, 0], atol=0.02), f"bad frequencies {frequencies}"

def test_measure_single_basis_state():
    sv = StateVector(2, state=[0, 0, 1, 0])
    assert sv.measure_single(0) == 0, "qubit 0 of cell 2 should be 0"
    assert sv.measure_single(1) == 1, "qubit 1 of cell 2 should be 1"


def test_measure_single_collapse():
    rng = np.random.default_rng(42)
    for _ in range(100):
        sv = StateVector(2, state=[0.6, 0, 0.8, 0], rng=rng)
        result = sv.measure_single(1)
        if result == 0:
            expected_state = [1, 0, 0, 0]
        else:
            expected_state = [0, 0, 1, 0]
        assert np.allclose(sv.state, expected_state), f"wrong collapse for result {result}"
        assert np.isclose(np.linalg.norm(sv.state), 1), "state not normalized after measurement"


def test_measure_single_statistics():
    rng = np.random.default_rng(42)
    shots = 10000
    ones = 0
    for _ in range(shots):
        sv = StateVector(2, state=[0.6, 0, 0.8, 0], rng=rng)
        ones += sv.measure_single(1)
    frequency = ones / shots
    assert abs(frequency - 0.64) < 0.02, f"bad frequency {frequency}"


def test_measure_single_entangled():
    rng = np.random.default_rng(42)
    for _ in range(100):
        sv = StateVector(2, state=[1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], rng=rng)
        a = sv.measure_single(0)
        b = sv.measure_single(1)
        assert a == b, f"entangled qubits gave different results: {a}, {b}"