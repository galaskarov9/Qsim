import numpy as np

from qsim.state_vector import StateVector
from qsim.algorithms.oracles import make_oracle_matrix
from qsim.algorithms.grover import diffuser


def uniform_state(n):
    sv = StateVector(n)
    sv.state = np.full(2**n, 1 / np.sqrt(2**n), dtype=complex)
    return sv


def test_diffuser_exact_result_on_basis_state():
    sv = StateVector(2)
    sv.state = np.array([1, 0, 0, 0], dtype=complex)
    result = diffuser(sv)

    expected = np.array([0.5, -0.5, -0.5, -0.5], dtype=complex)
    assert np.allclose(result.state, expected), \
        "gate diffuser on 00 should give minus (2|s><s| - I)|00>"


def test_diffuser_keeps_uniform_superposition_up_to_global_phase():
    result = diffuser(uniform_state(3))
    probabilities = np.abs(result.state) ** 2
    assert np.allclose(probabilities, np.full(8, 1 / 8)), \
        "reflection around |s> should leave |s> unchanged (up to a global sign)"


def test_one_grover_iteration_finds_marked_state_with_two_qubits():
    sv = uniform_state(2)
    sv.apply_full_matrix(make_oracle_matrix(lambda x: x == 3, 2))
    result = diffuser(sv)

    probabilities = np.abs(result.state) ** 2
    assert np.allclose(probabilities, [0, 0, 0, 1]), \
        "with 2 qubits, one Grover iteration should find 11 with probability 1"


def test_diffuser_does_not_change_input_state():
    sv = uniform_state(2)
    sv.apply_full_matrix(make_oracle_matrix(lambda x: x == 3, 2))
    before = sv.state.copy()

    diffuser(sv)
    assert np.allclose(sv.state, before), \
        "diffuser should return a new state and leave the input unchanged"