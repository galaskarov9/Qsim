import numpy as np

def make_oracle_matrix(f, n):
    return np.diag([-1 if f(x) else 1 for x in range(2**n)])