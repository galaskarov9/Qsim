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
c.add_gate(X, 0, "CNOT", controls=1)

print(c)
print()
print("Final state")
print_state(c.apply_circuit())
