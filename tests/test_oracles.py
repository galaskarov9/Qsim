import numpy as np

from qsim.algorithms.oracles import make_oracle_matrix
from qsim.state_vector import StateVector


def test_oracle_single_marked_state():
    oracle = make_oracle_matrix(lambda x: x == 3, 2)
    expected = np.diag([1, 1, 1, -1])
    assert np.array_equal(oracle, expected), "only the marked state (index 3) should get -1 on the diagonal"


def test_oracle_multiple_marked_states():
    oracle = make_oracle_matrix(lambda x: x in (1, 6), 3)
    expected = np.diag([1, -1, 1, 1, 1, 1, -1, 1])
    assert np.array_equal(oracle, expected), "every marked state (indices 1 and 6) should get -1, all others +1"


def test_oracle_with_no_marked_states_is_identity():
    oracle = make_oracle_matrix(lambda x: False, 3)
    assert np.array_equal(oracle, np.eye(8)), "with no marked states the oracle should be the identity"


def test_oracle_shape():
    oracle = make_oracle_matrix(lambda x: x == 0, 4)
    assert oracle.shape == (16, 16), f"4 qubits should give a 16x16 matrix, got {oracle.shape}"


def test_oracle_is_its_own_inverse():
    oracle = make_oracle_matrix(lambda x: x % 2 == 0, 3)
    assert np.allclose(oracle @ oracle, np.eye(8)), "applying the oracle twice should give the identity"


def test_oracle_applied_to_state_vector():
    sv = StateVector(2)
    sv.state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)

    sv.apply_full_matrix(make_oracle_matrix(lambda x: x == 3, 2))

    expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
    assert np.allclose(sv.state, expected), "oracle on uniform superposition should flip only the amplitude of 11"