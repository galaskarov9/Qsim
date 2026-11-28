import numpy as np
import pytest
from qsim.QuantumCircuit import QuantumCircuit
from qsim.state_vector import StateVector
from qsim.algorithms.qft import qft, inverse_qft


def basis_state(n, x):
    sv = StateVector(n)
    sv.state = np.zeros(2**n, dtype=complex)
    sv.state[x] = 1
    return sv


def run_qft_on_state(sv, qubits=None):
    c = QuantumCircuit(state_vector=sv)
    qft(c, list(range(sv.num_qubits)) if qubits is None else qubits)
    return c.apply_circuit().state


def run_qft(n, x, qubits=None):
    return run_qft_on_state(basis_state(n, x), qubits)


def dft_matrix(N):
    indices = np.arange(N)
    return np.exp(2j * np.pi * np.outer(indices, indices) / N) / np.sqrt(N)


def qft_matrix(n, qubits=None):
    return np.array([run_qft(n, x, qubits) for x in range(2**n)]).T


def test_qft_of_zero_state_is_uniform_superposition():
    result = run_qft(3, 0)
    assert np.allclose(result, np.full(8, 1 / np.sqrt(8))), \
        "QFT of |000> should be the uniform superposition"


def test_single_qubit_qft_is_hadamard():
    assert np.allclose(run_qft(1, 0), np.array([1, 1]) / np.sqrt(2)), \
        "QFT on one qubit should map |0> to |+>"
    assert np.allclose(run_qft(1, 1), np.array([1, -1]) / np.sqrt(2)), \
        "QFT on one qubit should map |1> to |->"


def test_two_qubit_qft_exact_result():
    expected = np.array([1, 1j, -1, -1j]) / 2
    assert np.allclose(run_qft(2, 1), expected), \
        "QFT of |01> should be 1/2 (1, i, -1, -i)"


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_dft_matrix(n):
    assert np.allclose(qft_matrix(n), dft_matrix(2**n)), \
        f"QFT circuit on {n} qubits should equal the DFT matrix"


def test_qft_is_unitary():
    M = qft_matrix(3)
    assert np.allclose(M.conj().T @ M, np.eye(8)), "QFT matrix should be unitary"


def test_qft_of_periodic_state():
    sv = StateVector(3)
    sv.state = np.zeros(8, dtype=complex)
    sv.state[[0, 2, 4, 6]] = 0.5
    result = run_qft_on_state(sv)

    expected = np.zeros(8, dtype=complex)
    expected[[0, 4]] = 1 / np.sqrt(2)
    assert np.allclose(result, expected), \
        "a state with period 2 on 8 values should transform to peaks at 0 and 4"


def test_qft_on_subset_of_qubits_leaves_other_qubits_alone():
    F4 = dft_matrix(4)
    one = np.array([0, 1], dtype=complex)
    for x_low in range(4):
        result = run_qft(3, 4 + x_low, qubits=[0, 1])
        expected = np.kron(one, F4[:, x_low])
        assert np.allclose(result, expected), \
            f"QFT on qubits 0,1 of |1{x_low:02b}> should keep qubit 2 at |1>"


def test_qubit_order_in_list_defines_bit_significance():
    swap = np.eye(4)[[0, 2, 1, 3]]
    expected = swap @ dft_matrix(4) @ swap
    assert np.allclose(qft_matrix(2, qubits=[1, 0]), expected), \
        "QFT on [1, 0] should equal the DFT with the qubits swapped"


def test_qft_gate_count():
    c = QuantumCircuit(4)
    qft(c, [0, 1, 2, 3])
    # 4 H + 6 controlled phases + 2 swaps * 3 CNOTs
    assert len(c.gates) == 16, "QFT on 4 qubits should add 16 gates"


def test_qft_on_empty_list_adds_no_gates():
    c = QuantumCircuit(2)
    qft(c, [])
    assert c.gates == [], "QFT on no qubits should add no gates"


def test_qft_raises_on_duplicate_qubits():
    c = QuantumCircuit(3)
    with pytest.raises(ValueError):
        qft(c, [0, 1, 1])


def test_qft_raises_on_out_of_range_qubit():
    c = QuantumCircuit(2)
    with pytest.raises(ValueError):
        qft(c, [0, 5])


def run_inverse_qft_on_state(sv, qubits=None):
    c = QuantumCircuit(state_vector=sv)
    inverse_qft(c, list(range(sv.num_qubits)) if qubits is None else qubits)
    return c.apply_circuit().state


def run_inverse_qft(n, x, qubits=None):
    return run_inverse_qft_on_state(basis_state(n, x), qubits)


def inverse_qft_matrix(n, qubits=None):
    return np.array([run_inverse_qft(n, x, qubits) for x in range(2**n)]).T


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_inverse_qft_matches_inverse_dft_matrix(n):
    expected = dft_matrix(2**n).conj().T
    assert np.allclose(inverse_qft_matrix(n), expected), \
        f"inverse QFT circuit on {n} qubits should equal the inverse DFT matrix"


def test_single_qubit_inverse_qft_is_hadamard():
    assert np.allclose(run_inverse_qft(1, 0), np.array([1, 1]) / np.sqrt(2)), \
        "inverse QFT on one qubit should map |0> to |+>"
    assert np.allclose(run_inverse_qft(1, 1), np.array([1, -1]) / np.sqrt(2)), \
        "inverse QFT on one qubit should map |1> to |->"


def test_two_qubit_inverse_qft_exact_result():
    expected = np.array([1, -1j, -1, 1j]) / 2
    assert np.allclose(run_inverse_qft(2, 1), expected), \
        "inverse QFT of |01> should be 1/2 (1, -i, -1, i)"


def test_inverse_qft_of_uniform_superposition_is_zero_state():
    sv = StateVector(3)
    sv.state = np.full(8, 1 / np.sqrt(8), dtype=complex)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1
    assert np.allclose(run_inverse_qft_on_state(sv), expected), \
        "inverse QFT of the uniform superposition should give |000>"


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_then_inverse_qft_is_identity(n):
    for x in range(2**n):
        c = QuantumCircuit(state_vector=basis_state(n, x))
        qft(c, list(range(n)))
        inverse_qft(c, list(range(n)))
        assert np.allclose(c.apply_circuit().state, basis_state(n, x).state), \
            f"QFT followed by inverse QFT should return |{x}> on {n} qubits"


def test_inverse_qft_then_qft_is_identity():
    rng = np.random.default_rng(7)
    sv = StateVector(3)
    amplitudes = rng.normal(size=8) + 1j * rng.normal(size=8)
    sv.state = amplitudes / np.linalg.norm(amplitudes)
    initial = sv.state.copy()

    c = QuantumCircuit(state_vector=sv)
    inverse_qft(c, [0, 1, 2])
    qft(c, [0, 1, 2])
    assert np.allclose(c.apply_circuit().state, initial), \
        "inverse QFT followed by QFT should return a random state unchanged"


def test_inverse_qft_on_subset_of_qubits_leaves_other_qubits_alone():
    F4_inv = dft_matrix(4).conj().T
    one = np.array([0, 1], dtype=complex)
    for x_low in range(4):
        result = run_inverse_qft(3, 4 + x_low, qubits=[0, 1])
        expected = np.kron(one, F4_inv[:, x_low])
        assert np.allclose(result, expected), \
            f"inverse QFT on qubits 0,1 of |1{x_low:02b}> should keep qubit 2 at |1>"


def test_inverse_qft_respects_qubit_order():
    for x in range(4):
        c = QuantumCircuit(state_vector=basis_state(2, x))
        qft(c, [1, 0])
        inverse_qft(c, [1, 0])
        assert np.allclose(c.apply_circuit().state, basis_state(2, x).state), \
            "inverse QFT on [1, 0] should undo QFT on [1, 0]"


def test_inverse_qft_gate_count_matches_qft():
    c_qft = QuantumCircuit(4)
    qft(c_qft, [0, 1, 2, 3])
    c_inv = QuantumCircuit(4)
    inverse_qft(c_inv, [0, 1, 2, 3])
    assert len(c_inv.gates) == len(c_qft.gates) == 16, \
        "inverse QFT should use the same number of gates as QFT"


def test_inverse_qft_on_empty_list_adds_no_gates():
    c = QuantumCircuit(2)
    inverse_qft(c, [])
    assert c.gates == [], "inverse QFT on no qubits should add no gates"


def test_inverse_qft_raises_on_duplicate_qubits():
    c = QuantumCircuit(3)
    with pytest.raises(ValueError):
        inverse_qft(c, [0, 1, 1])


def test_inverse_qft_raises_on_out_of_range_qubit():
    c = QuantumCircuit(2)
    with pytest.raises(ValueError):
        inverse_qft(c, [0, 5])
