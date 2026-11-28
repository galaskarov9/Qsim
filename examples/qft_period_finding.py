import math
import numpy as np
from qsim.algorithms.qft import qft
from qsim.state_vector import StateVector
from qsim.QuantumCircuit import QuantumCircuit

def periodic_state(n, r, offset):
    N = 2 ** n
    state = []
    for i in range(N):
        state.append(0)
    for i in range(offset, N, r):
        state[i] = np.sqrt(r / N)
    return state

def estimate_period(results, N):
    nonzero = [x for x in results if x != 0]
    if not nonzero:
        return None
    spacing = math.gcd(*nonzero)
    return N // spacing

def to_int_counts(counts):
    int_counts = {}
    for bitstring, count in counts.items():
        int_counts[int(bitstring, 2)] = count
    return dict(sorted(int_counts.items()))


def format_counts(counts):
    return "  ".join(f"{value:>2}: {count:<4}" for value, count in counts.items())

n = 4
r = 4
offsets = [0, 1, 2]
qubits = [0, 1, 2, 3]
rng = np.random.default_rng(seed=42)

N = 2 ** n

print("=" * 60)
print(f"QFT on a periodic state: period r = {r}, N = {N}")
print("=" * 60)
print(f"Expected peaks after the QFT: multiples of N / r = {N // r}")

for offset in offsets:
    sv = StateVector(n, periodic_state(n, r, offset), rng)
    empty_circuit = QuantumCircuit(state_vector=sv)
    qft_circuit = QuantumCircuit(state_vector=sv)
    qft(qft_circuit, qubits)
    empty_sample = empty_circuit.sample(1000)
    qft_sample = qft_circuit.sample(1000)
    print()
    print(f"Offset {offset}:")
    print(f"  Without QFT: {format_counts(to_int_counts(empty_sample))}")
    print(f"  With QFT:    {format_counts(to_int_counts(qft_sample))}")

runs = 1000
empty_totals = {}
qft_totals = {}
qft_results = []

for _ in range(runs):
    offset = int(rng.integers(0, r))
    sv = StateVector(n, periodic_state(n, r, offset), rng)
    empty_circuit = QuantumCircuit(state_vector=sv)
    qft_circuit = QuantumCircuit(state_vector=sv)
    qft(qft_circuit, qubits)

    empty_result = next(iter(empty_circuit.sample(1)))
    qft_result = next(iter(qft_circuit.sample(1)))

    empty_totals[empty_result] = empty_totals.get(empty_result, 0) + 1
    qft_totals[qft_result] = qft_totals.get(qft_result, 0) + 1
    qft_results.append(int(qft_result, 2))

empty_counts = to_int_counts(empty_totals)
qft_counts = to_int_counts(qft_totals)

print()
print("=" * 60)
print(f"Realistic experiment: {runs} runs, each with a random offset")
print("and a single measurement")
print("=" * 60)
print(f"{'Result':>6} | {'Without QFT':>11} | {'With QFT':>8}")
print("-" * 33)
for value in range(N):
    print(f"{value:>6} | {empty_counts.get(value, 0):>11} | {qft_counts.get(value, 0):>8}")

print()
print("=" * 60)
print("Recovering the period from the first few measurements")
print("=" * 60)
for shots in [1, 2, 3, 5, 10]:
    first = qft_results[:shots]
    estimate = estimate_period(first, N)
    status = "unknown" if estimate is None else ("correct" if estimate == r else "wrong")
    print(f"{shots:>2} measurements {str(first):<40} -> r = {estimate} ({status})")
print(f"True period: r = {r}")