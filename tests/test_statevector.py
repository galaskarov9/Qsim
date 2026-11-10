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