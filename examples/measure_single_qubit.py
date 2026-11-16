import numpy as np
from qsim.QuantumCircuit import QuantumCircuit
from qsim.gates import X, H

def print_state(sv):
    for i, amp in enumerate(sv.state):
        if not np.isclose(amp, 0):
            bits = format(i, f'0{sv.num_qubits}b')
            print(f"|{bits}>: {amp}")

c = QuantumCircuit(2)
c.add_gate(H, 1, "H")
c.add_gate(X, 0, "CNOT", control=1)

for run in range(3):
    sv = c.apply_circuit()
    print(f"Run {run + 1}")
    print("Before measurement:")
    print_state(sv)

    result_0 = sv.measure_single(0)
    print(f"Measured qubit 0: {result_0}")
    print("After measuring qubit 0:")
    print_state(sv)

    result_1 = sv.measure_single(1)
    print(f"Measured qubit 1: {result_1}")
    print()