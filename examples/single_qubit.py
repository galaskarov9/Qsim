import numpy as np
from qsim.qubit import Qubit
from qsim.gates import H

#Part 1: Superposition and collapse example

qubit = Qubit()
qubit.apply_gate(H)
print(qubit.state)
print(qubit.probabilities())
for _ in range(10):
    result = qubit.measure()
    print(f"Measurement result: {result}, New state: {qubit.state}")
print("Measuring the same qubit 10 times. After the first measurement the state collapses, so all results are the same:")

#Part 2: Statistics over many qubits

rng = np.random.default_rng(seed=42)
zero_count = 0
for _ in range(1000):
    qubit =Qubit(rng=rng)
    qubit.apply_gate(H)
    result = qubit.measure()
    if result == 0:
        zero_count += 1
print(f"Number of 0s measured: {zero_count}")  # Expect roughly half to be 0