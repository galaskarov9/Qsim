import numpy as np
from qsim.gates import X, H

def test_gates():
    assert np.allclose(X@X, np.eye(2, dtype=complex))  # X gate squared is identity
    assert np.allclose(H@H, np.eye(2, dtype=complex))  # H gate squared is identity

def test_gate_unitarity():
    assert np.allclose(X@X.conj().T, np.eye(2, dtype=complex))  # X gate is unitary
    assert np.allclose(H@H.conj().T, np.eye(2, dtype=complex))  # H gate is unitary

def test_gate_size():
    assert X.shape == (2, 2)
    assert H.shape == (2, 2)