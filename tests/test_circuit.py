import numpy as np
from qsim.QuantumCircuit import QuantumCircuit
from qsim.state_vector import StateVector
from qsim.gates import X, H


def test_circuit_runs_gates_in_order():
    c = QuantumCircuit(2)
    c.add_gate(H, 0, "H")
    c.add_gate(X, 1, "X")
    sv = c.apply_circuit()
    expected_state = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(sv.state, expected_state), "wrong state after H on 0 and X on 1"


def test_circuit_runs_twice_same_result():
    c = QuantumCircuit(1)
    c.add_gate(X, 0, "X")
    first = c.apply_circuit()
    second = c.apply_circuit()
    assert np.allclose(first.state, [0, 1]), "first run is wrong"
    assert np.allclose(second.state, [0, 1]), "second run should give the same result"


def test_circuit_does_not_change_given_state():
    initial = StateVector(1)
    c = QuantumCircuit(state_vector=initial)
    c.add_gate(X, 0, "X")
    c.apply_circuit()
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
    sv = c.apply_circuit()
    expected_state = np.full(1024, 1/32, dtype=complex)
    assert np.allclose(sv.state, expected_state), "H on all qubits should give uniform state"

def test_circuit_with_controlled_gate():
    c = QuantumCircuit(2)
    c.add_gate(H, 1, "H")
    c.add_gate(X, 0, "CNOT", controls=1)
    sv = c.apply_circuit()
    expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(sv.state, expected_state), "circuit should create entangled state"


def test_circuit_invalid_control():
    c = QuantumCircuit(2)
    try:
        c.add_gate(X, 0, "CNOT", controls=0)
        assert False, "same control and target should raise an error"
    except ValueError:
        pass


def test_circuit_print_with_control():
    c = QuantumCircuit(2)
    c.add_gate(X, 0, "CNOT", controls=1)
    assert "controlled by qubit 1" in str(c), "print should show the control qubit"


def test_ten_qubit_entangled_chain():
    c = QuantumCircuit(10)
    c.add_gate(H, 0, "H")
    for q in range(9):
        c.add_gate(X, q + 1, "CNOT", controls=q)
    sv = c.apply_circuit()
    expected_state = np.zeros(1024, dtype=complex)
    expected_state[0] = 1/np.sqrt(2)
    expected_state[1023] = 1/np.sqrt(2)
    assert np.allclose(sv.state, expected_state), "chain should give all zeros or all ones"

def test_swap():
    c = QuantumCircuit(2)
    c.add_gate(X, 0, "X")
    c.add_swap_gate(0, 1)
    sv = c.apply_circuit()
    assert np.allclose(sv.state, [0, 0, 1, 0]), "SWAP should move 01 to 10"

def test_circuit_with_toffoli_flips_target_when_controls_are_one():
    c = QuantumCircuit(3)
    c.add_gate(X, 0, "X")
    c.add_gate(X, 1, "X")
    c.add_gate(X, 2, "Toffoli", controls=[0, 1])
    sv = c.apply_circuit()

    expected = np.zeros(8, dtype=complex)
    expected[7] = 1
    assert np.allclose(sv.state, expected), \
        "both controls are 1, so Toffoli should flip target: 011 -> 111"


def test_circuit_with_toffoli_does_nothing_when_one_control_is_zero():
    c = QuantumCircuit(3)
    c.add_gate(X, 0, "X")
    c.add_gate(X, 2, "Toffoli", controls=[0, 1])
    sv = c.apply_circuit()

    expected = np.zeros(8, dtype=complex)
    expected[1] = 1
    assert np.allclose(sv.state, expected), \
        "control 1 is 0, so Toffoli should do nothing: 001 stays 001"


def test_circuit_print_with_multiple_controls():
    c = QuantumCircuit(3)
    c.add_gate(X, 2, "Toffoli", controls=[0, 1])
    assert "controlled by qubits 0, 1" in str(c), \
        "print should list all control qubits"


def test_circuit_print_without_control():
    c = QuantumCircuit(2)
    c.add_gate(X, 0, "X")
    assert "controlled" not in str(c), \
        "a gate with no controls should not mention controls"