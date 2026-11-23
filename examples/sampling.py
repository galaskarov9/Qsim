import numpy as np
from qsim.QuantumCircuit import QuantumCircuit
from qsim.gates import H, X


c = QuantumCircuit(3)
c.add_gate(H, 0, "H")
c.add_gate(X, 1, "CNOT", controls=0)
c.add_gate(X, 2, "CNOT", controls=1)

print(c)
print()

shots = 1000
counts = c.sample(shots)
probabilities = c.apply_circuit().probabilities()

print(f"Results of {shots} shots:")
for index, p in enumerate(probabilities):
    bits = format(index, "03b")
    count = counts.get(bits, 0)
    print(f"  |{bits}>  count {count:4d}   expected {p:.3f}")