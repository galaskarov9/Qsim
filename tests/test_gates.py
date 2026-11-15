import numpy as np
from qsim.state_vector import StateVector
from qsim.gates import I, X, Y, Z, H, S, T, P, Rx, Ry, Rz


def is_unitary(m):
    return np.allclose(m @ m.conj().T, I)


def test_fixed_gates_unitary():
    for name, gate in [("I", I), ("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T)]:
        assert is_unitary(gate), f"{name} is not unitary"


def test_rotation_gates_unitary():
    for theta in [0, 0.3, np.pi / 2, np.pi, 2.5, 2 * np.pi]:
        for name, gate in [("P", P), ("Rx", Rx), ("Ry", Ry), ("Rz", Rz)]:
            assert is_unitary(gate(theta)), f"{name}({theta}) is not unitary"


def test_gate_relations():
    assert np.allclose(T @ T, S), "T·T should equal S"
    assert np.allclose(S @ S, Z), "S·S should equal Z"
    assert np.allclose(P(np.pi / 2), S), "P(π/2) should equal S"
    assert np.allclose(P(np.pi / 4), T), "P(π/4) should equal T"


def test_ry_pi_flips_zero_to_one():
    sv = StateVector(1)
    sv.apply_gate(Ry(np.pi), 0)
    assert np.allclose(sv.probabilities(), [0, 1]), "Ry(π) should take 0 to 1"


def test_interference():
    sv = StateVector(1)
    sv.apply_gate(H, 0)
    sv.apply_gate(S, 0)
    sv.apply_gate(S, 0)
    sv.apply_gate(H, 0)
    assert np.allclose(sv.probabilities(), [0, 1]), "H S S H should give 1 with certainty"