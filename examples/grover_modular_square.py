import numpy as np
from qsim.algorithms.grover import grover

n = 4
N = 2 ** n


def f(x):
    return x ** 2 % 16 == 9


M = 0
for x in range(N):
    if f(x):
        M += 1

theta = np.arcsin(np.sqrt(M / N))
k = int(np.floor(np.pi / (4 * theta)))

sv = grover(f, n)
probabilities = sv.probabilities()

print("=" * 50)
print(f"Grover's search: find an x between 0 and {N - 1} with x^2 mod 16 == 9")
print("=" * 50)
print(f"Qubits: {n} | States: {N} | Solutions: {M} | Iterations: {k}")
print()

success = sum(probabilities[x] for x in range(N) if f(x))
print(f"Probability of measuring a solution: {success:.4f}")
print("Likely results:")
for x in range(N):
    if probabilities[x] > 0.01:
        print(f"  {x:0{n}b} ({x}) with probability {probabilities[x]:.4f}")
print(f"Classical search: about {(N + 1) / (M + 1):.1f} checks on average | "
      f"Grover: {k} oracle queries")