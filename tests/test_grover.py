import numpy as np
import pytest
from qsim.state_vector import StateVector
from qsim.algorithms.oracles import make_oracle_matrix
from qsim.algorithms.grover import diffuser, grover


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

def probabilities(sv):
    return np.abs(sv.state) ** 2


def test_grover_two_qubits_finds_marked_state_with_certainty():
    sv = grover(lambda x: x == 3, 2)
    assert np.allclose(probabilities(sv), [0, 0, 0, 1]), \
        "with 2 qubits and one solution, Grover should find it with probability 1"


def test_grover_three_qubits_finds_marked_state_with_high_probability():
    sv = grover(lambda x: x == 5, 3)
    probs = probabilities(sv)
    assert np.argmax(probs) == 5, "the marked state should be the most likely result"
    assert np.isclose(probs[5], 0.9453, atol=1e-3), \
        "with 3 qubits, 2 iterations should give probability of about 0.945"


def test_grover_multiple_solutions():
    sv = grover(lambda x: x in (1, 6), 3)
    probs = probabilities(sv)
    assert np.isclose(probs[1] + probs[6], 1), \
        "with 2 solutions out of 8, one iteration should find a solution with certainty"
    assert np.isclose(probs[1], probs[6]), "both solutions should be equally likely"


def test_grover_too_many_iterations_overshoots():
    sv = grover(lambda x: x == 3, 2, k=2)
    assert np.isclose(probabilities(sv)[3], 0.25), \
        "with 2 qubits, 2 iterations should overshoot and drop the probability to 0.25"


def test_grover_more_than_half_solutions_does_no_iterations():
    sv = grover(lambda x: x != 0, 2)
    assert np.allclose(probabilities(sv), np.full(4, 0.25)), \
        "with 3 solutions out of 4, zero iterations should leave the uniform superposition"


def test_grover_raises_when_there_are_no_solutions():
    with pytest.raises(ValueError):
        grover(lambda x: False, 3)


def test_grover_raises_on_invalid_num_qubits():
    with pytest.raises(ValueError):
        grover(lambda x: x == 0, 0)
    with pytest.raises(ValueError):
        grover(lambda x: x == 0, 2.5)


def test_grover_raises_on_invalid_k():
    with pytest.raises(ValueError):
        grover(lambda x: x == 0, 2, k=-1)
    with pytest.raises(ValueError):
        grover(lambda x: x == 0, 2, k=1.5)