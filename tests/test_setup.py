import numpy as np
import qsim
def test_numpy_works():
    assert np.array([1, 2]).sum() == 3
def test_qsim_works():
    assert qsim is not None