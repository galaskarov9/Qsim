# qsim — a quantum computer simulator

A state-vector quantum computer simulator written in Python with NumPy, built without any quantum computing libraries.

## Features
- **State Vector** for any number of qubits, stored as a single complex vector of length 2^n
- **Single-qubit gates:** X, Y, Z, H, S, T, phase P(θ), and rotations Rx(θ), Ry(θ), Rz(θ)
- **Controlled gates:** any single-qubit gate can be applied with a control qubit
- **Measurement:** of a single qubit (partial collapse with renormalization) or of the whole State Vector
- **Quantum circuits:** build a circuit as a list of gates, run it, print it
- **Sampling:** run a circuit for many shots and get counts, as on real quantum hardware
- **Reproducible randomness:** every random operation uses an injectable NumPy random generator

Gates are applied directly to pairs of amplitudes rather than by building full 2^n × 2^n matrices, so memory and time stay linear in the size of the state vector.

## Qubit ordering

Qubit 0 is the rightmost (least significant) bit of a basis-state index. For example, with 3 qubits, the basis state `|011>` (index 3) has qubits 0 and 1 in state 1 and qubit 2 in state 0.

## Quick start
```python
from qsim.circuit import QuantumCircuit
from qsim.gates import H, X

# Bell state: (|00> + |11>) / √2
c = QuantumCircuit(2)
c.add_gate(H, 1, "H")
c.add_gate(X, 0, "CNOT", control=1)

print(c)
print(c.sample(1000))   # e.g. {'00': 503, '11': 497}
```

## Project structure
```
qsim/
  state_vector.py   state vector, gate application, measurement
  QuantumCircuit.py        quantum circuit, running and sampling
  gates.py          gate matrices and parametrized gates
  qubit.py
tests/              unit tests
examples/           runnable demonstrations
```
## Examples
Run from the project root:
```
python -m examples.01_build_circuit
python -m examples.02_measure_single_qubit
python -m examples.03_measure_state_vector
python -m examples.04_sampling
```

| Example | What it shows |
|---|---|
| 01 | Building a circuit that creates an entangled state |
| 02 | Measuring one qubit of an entangled pair determines the other |
| 03 | Full measurement of a uniform superposition collapses to a single basis state |
| 04 | Sampling a 3-qubit entangled state and comparing counts to exact probabilities |

## Testing
```
pip install numpy pytest
pytest
```

The tests include checks against independent computations: gate application on 10 qubits is compared to the tensor product of single-qubit states, and controlled gates are compared to full matrices built from projectors for every control/target pair.