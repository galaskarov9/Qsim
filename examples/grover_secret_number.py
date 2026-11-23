import numpy as np
from qsim.algorithms.grover import grover

secret_number = 11
n = 4
N = 2 ** n

def f(x):
    return x == secret_number

M = 0
for x in range(N):
    if f(x):
        M += 1    

theta = np.arcsin(np.sqrt(M / N))
k = int(np.floor(np.pi / (4 * theta)))

sv = grover(f, n)
probabilities = sv.probabilities()

success = sum(probabilities[x] for x in range(N) if f(x))
most_likely = int(np.argmax(probabilities))

print("=" * 50)
print(f"Grover's search: find a secret number between 0 and {N - 1}")
print("=" * 50)
print(f"Qubits: {n} | States: {N} | Solutions: {M} | Iterations: {k}")
print()

print(f"Probability of measuring a solution: {success:.4f}")
print(f"Most likely result: {most_likely:0{n}b} ({most_likely}) "
      f"with probability {probabilities[most_likely]:.4f}")
print(f"Classical search: about {N // 2} checks on average | "
      f"Grover: {k} oracle queries")