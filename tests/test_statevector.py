import numpy as np
import pytest
from qsim.state_vector import StateVector
from qsim.gates import X, H, Y, Z, S, T, Rx, Ry, I

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

def test_controlled_x_control_zero():
    sv = StateVector(2)
    sv.apply_controlled_gate(X, 1, 0)
    assert np.allclose(sv.state, [1, 0, 0, 0]), "control is 0, nothing should change"


def test_controlled_x_control_one():
    sv = StateVector(2, state=[0, 0, 1, 0])
    sv.apply_controlled_gate(X, 1, 0)
    assert np.allclose(sv.state, [0, 0, 0, 1]), "control is 1, target should flip"


def test_controlled_x_reversed_roles():
    sv = StateVector(2, state=[0, 1, 0, 0])
    sv.apply_controlled_gate(X, 0, 1)
    assert np.allclose(sv.state, [0, 0, 0, 1]), "qubit 0 controls, qubit 1 should flip"


def test_controlled_x_creates_entanglement():
    sv = StateVector(2)
    sv.apply_gate(H, 1)
    sv.apply_controlled_gate(X, 1, 0)
    expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(sv.state, expected_state), "H then controlled X should give entangled state"


def test_controlled_gate_invalid_qubits():
    sv = StateVector(2)
    for control, target in [(1, 1), (2, 0), (0, 2), (-1, 0)]:
        try:
            sv.apply_controlled_gate(X, control, target)
            assert False, f"control {control}, target {target} should raise an error"
        except ValueError:
            pass

def test_controlled_x_far_qubits():
    sv = StateVector(4)
    sv.apply_gate(X, 0)
    sv.apply_gate(X, 3)
    sv.apply_controlled_gate(X, 3, 1)
    expected_state = np.zeros(16, dtype=complex)
    expected_state[11] = 1
    assert np.allclose(sv.state, expected_state), "cell 9 should move to cell 11"

def test_controlled_h():
    sv = StateVector(2, state=[0, 0, 1, 0])
    sv.apply_controlled_gate(H, 1, 0)
    expected_state = np.array([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(sv.state, expected_state), "controlled H with control 1 should act on target"

def test_apply_full_matrix_flips_sign_of_marked_state():
    sv = StateVector(2)
    sv.state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    oracle = np.diag([1, 1, 1, -1])
    sv.apply_full_matrix(oracle)

    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    assert np.allclose(sv.state, expected)


def test_apply_full_matrix_raises_on_wrong_shape():
    sv = StateVector(2)
    wrong_matrix = np.eye(2)

    with pytest.raises(ValueError):
        sv.apply_full_matrix(wrong_matrix)

def basis_state(n, index):
    sv = StateVector(n)
    sv.state = np.zeros(2**n, dtype=complex)
    sv.state[index] = 1
    return sv

def test_multi_controlled_with_no_controls_acts_like_single_gate():
    sv = basis_state(2, 0)
    sv.apply_multi_controlled_gate(X, [], 1)
    assert np.allclose(sv.state, basis_state(2, 2).state), \
        "with no controls, X on qubit 1 should move 00 to 10"


def test_multi_controlled_with_one_control_acts_like_cnot():
    sv = basis_state(2, 1)
    sv.apply_multi_controlled_gate(X, [0], 1)
    assert np.allclose(sv.state, basis_state(2, 3).state), \
        "control 0 is 1, so target 1 should flip: 01 -> 11"


def test_toffoli_flips_target_when_both_controls_are_one():
    sv = basis_state(3, 3)
    sv.apply_multi_controlled_gate(X, [0, 1], 2)
    assert np.allclose(sv.state, basis_state(3, 7).state), \
        "controls 0 and 1 are both 1, so target 2 should flip: 011 -> 111"


def test_toffoli_does_nothing_when_one_control_is_zero():
    sv = basis_state(3, 1)
    sv.apply_multi_controlled_gate(X, [0, 1], 2)
    assert np.allclose(sv.state, basis_state(3, 1).state), \
        "control 1 is 0, so nothing should change: 001 stays 001"


def test_multi_controlled_with_non_adjacent_qubits():
    sv = basis_state(4, 9)
    sv.apply_multi_controlled_gate(X, [0, 3], 1)
    assert np.allclose(sv.state, basis_state(4, 11).state), \
        "controls 0 and 3 are 1, so target 1 should flip: 1001 -> 1011"


def test_multi_controlled_z_flips_only_last_state():
    sv = StateVector(3)
    sv.state = np.full(8, 1 / np.sqrt(8), dtype=complex)
    sv.apply_multi_controlled_gate(Z, [0, 1], 2)

    expected = np.full(8, 1 / np.sqrt(8), dtype=complex)
    expected[7] *= -1
    assert np.allclose(sv.state, expected), \
        "multi-controlled Z should flip only the sign of 111"


def test_multi_controlled_raises_on_wrong_gate_shape():
    sv = StateVector(3)
    with pytest.raises(ValueError):
        sv.apply_multi_controlled_gate(np.eye(4), [0], 1)


def test_multi_controlled_raises_when_target_is_also_control():
    sv = StateVector(3)
    with pytest.raises(ValueError):
        sv.apply_multi_controlled_gate(X, [0, 2], 2)


def test_multi_controlled_raises_on_duplicate_controls():
    sv = StateVector(3)
    with pytest.raises(ValueError):
        sv.apply_multi_controlled_gate(X, [0, 0], 2)


def test_multi_controlled_raises_on_out_of_range_qubit():
    sv = StateVector(3)
    with pytest.raises(ValueError):
        sv.apply_multi_controlled_gate(X, [0, 5], 2)