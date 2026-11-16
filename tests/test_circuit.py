import numpy as np
from qsim.QuantumCircuit import QuantumCircuit
from qsim.state_vector import StateVector
from qsim.gates import X, H


def test_circuit_runs_gates_in_order():
    c = QuantumCircuit(2)
    c.add_gate(H, 0, "H")
    c.add_gate(X, 1, "X")
    sv = c.apply_gates()
    expected_state = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(sv.state, expected_state), "wrong state after H on 0 and X on 1"


def test_circuit_runs_twice_same_result():
    c = QuantumCircuit(1)
    c.add_gate(X, 0, "X")
    first = c.apply_gates()
    second = c.apply_gates()
    assert np.allclose(first.state, [0, 1]), "first run is wrong"
    assert np.allclose(second.state, [0, 1]), "second run should give the same result"


def test_circuit_does_not_change_given_state():
    initial = StateVector(1)
    c = QuantumCircuit(state_vector=initial)
    c.add_gate(X, 0, "X")
    c.apply_gates()
    assert np.allclose(initial.state, [1, 0]), "circuit changed the state vector it was given"


def test_add_gate_invalid_target():
    c = QuantumCircuit(2)
    try:
        c.add_gate(X, 2, "X")
        assert False, "adding a gate on a missing qubit should raise an error"
    except ValueError:
        pass

def test_ten_qubits_h_on_all():
    c = QuantumCircuit(10)
    for q in range(10):
        c.add_gate(H, q, "H")
    sv = c.apply_gates()
    expected_state = np.full(1024, 1/32, dtype=complex)
    assert np.allclose(sv.state, expected_state), "H on all qubits should give uniform state"