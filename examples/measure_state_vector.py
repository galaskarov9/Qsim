import numpy as np
from qsim.QuantumCircuit import QuantumCircuit
from qsim.gates import H


def print_state(sv):
    for index, amplitude in enumerate(sv.state):
        if not np.isclose(amplitude, 0):
            bits = format(index, f"0{sv.num_qubits}b")
            print(f"  |{bits}>  {amplitude:.3f}")


c = QuantumCircuit(3)
for q in range(3):
    c.add_gate(H, q, "H")

sv = c.apply_circuit()
print("Before measurement:")
print_state(sv)
print()

for run in range(5):
    sv = c.apply_circuit()
    index = sv.measure()
    bits = format(index, "03b")
    print(f"Run {run + 1}: measured |{bits}>, state collapsed to:")
    print_state(sv)