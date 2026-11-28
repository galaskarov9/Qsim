# qsim — a quantum computer simulator

A state-vector quantum computer simulator written in Python with NumPy, built without any quantum computing libraries.

## Features
- **State Vector** for any number of qubits, stored as a single complex vector of length 2^n
- **Single-qubit gates:** X, Y, Z, H, S, T, phase P(θ), and rotations Rx(θ), Ry(θ), Rz(θ)
- **Controlled gates:** any single-qubit gate can be applied with one or more control qubits (e.g. CNOT, Toffoli, multi-controlled Z)
- **Measurement:** of a single qubit (partial collapse with renormalization) or of the whole State Vector
- **Quantum circuits:** build a circuit as a list of gates, run it, print it
- **Sampling:** run a circuit for many shots and get counts, as on real quantum hardware
- **Reproducible randomness:** every random operation uses an injectable NumPy random generator
- **Grover's search algorithm:** find marked inputs of any boolean function `f(x)` with roughly √N oracle queries
- **Quantum Fourier transform:** QFT and inverse QFT on any list of qubits in a circuit

Gates are applied directly to pairs of amplitudes rather than by building full 2^n × 2^n matrices, so memory and time stay linear in the size of the state vector.

## Qubit ordering

Qubit 0 is the rightmost (least significant) bit of a basis-state index. For example, with 3 qubits, the basis state `|011>` (index 3) has qubits 0 and 1 in state 1 and qubit 2 in state 0.

## Quick start
```python
from qsim.QuantumCircuit import QuantumCircuit
from qsim.gates import H, X

# Bell state: (|00> + |11>) / √2
c = QuantumCircuit(2)
c.add_gate(H, 1, "H")
c.add_gate(X, 0, "CNOT", controls=1)

print(c)
print(c.sample(1000))   # e.g. {'00': 503, '11': 497}
```

## Grover's algorithm
`grover(f, num_qubits, k=None)` searches the `2^num_qubits` inputs for those where `f(x)` is `True`:

1. Put every qubit in superposition with H.
2. Repeat `k` times: apply the **oracle** (flips the sign of every marked state) and then the **diffuser** (reflects the state around the uniform superposition).
3. Return the final state vector. Marked states now have a high probability of being measured.

If `k` is not given, the optimal number of iterations is used: `floor(π / (4θ))` where `θ = arcsin(√(M/N))`, `M` is the number of solutions and `N = 2^num_qubits`. A `ValueError` is raised if `f` has no solutions in range.

```python
from qsim.algorithms.grover import grover

sv = grover(lambda x: x == 5, 3)   # find 5 among 0..7
print(sv.probabilities().round(3))
# [0.008 0.008 0.008 0.008 0.008 0.945 0.008 0.008]
```

## Quantum Fourier transform
`qft(circuit, qubits_list)` adds the gates of the quantum Fourier transform to a circuit, acting on the qubits in `qubits_list`. `inverse_qft(circuit, qubits_list)` adds the gates of its inverse.

For `n` qubits and `N = 2^n`, the QFT maps each basis state `|x>` to

```
|x>  ->  1/√N · Σ_y e^(2πi·xy/N) |y>
```

The circuit is built from H gates, controlled phase gates P(π/2^k), and SWAPs that reverse the qubit order at the end. The inverse applies the same gates in reverse order with negated phases.

- The first qubit in `qubits_list` is treated as the least significant bit of `x`, so the order of the list matters.
- Qubits not in the list are left unchanged, so the QFT can run on part of a larger register.
- A `ValueError` is raised if `qubits_list` contains a qubit twice or a qubit outside the circuit.

```python
from qsim.QuantumCircuit import QuantumCircuit
from qsim.algorithms.qft import qft, inverse_qft
from qsim.gates import X

c = QuantumCircuit(3)
c.add_gate(X, 0, "X")       # start in |001>
qft(c, [0, 1, 2])
inverse_qft(c, [0, 1, 2])   # undo the QFT
print(c.apply_circuit().probabilities().round(3))
# [0. 1. 0. 0. 0. 0. 0. 0.]
```

## Project structure
```
qsim/
  state_vector.py      state vector, gate application, measurement
  QuantumCircuit.py    quantum circuit, running and sampling
  gates.py             gate matrices and parametrized gates
  qubit.py
  algorithms/
    grover.py          Grover's search and the diffuser
    oracles.py         phase oracle built from a boolean function
    qft.py             quantum Fourier transform and its inverse
tests/                 unit tests
examples/              runnable demonstrations
```
## Examples
Run from the project root:
```
python -m examples.build_circuit
python -m examples.measure_single_qubit
python -m examples.measure_state_vector
python -m examples.sampling
python -m examples.grover_secret_number
python -m examples.grover_modular_square
python -m examples.qft_period_finding
```

| Example | What it shows |
|---|---|
| build_circuit | Building a circuit that creates an entangled state |
| measure_single_qubit | Measuring one qubit of an entangled pair determines the other |
| measure_state_vector | Full measurement of a uniform superposition collapses to a single basis state |
| sampling | Sampling a 3-qubit entangled state and comparing counts to exact probabilities |
| grover_secret_number | Grover's search for one secret number among 16: after 3 iterations it is measured with probability ≈ 0.96 |
| grover_modular_square | Grover's search with several solutions: the 4 values of x with x² mod 16 = 9, found with certainty after 1 iteration |
| qft_period_finding | The QFT turns a periodic state with period 4 into peaks at multiples of 16/4, whatever the offset, and the period is recovered from a few measurements |

## Testing
```
pip install numpy pytest
pytest
```

The tests include checks against independent computations: gate application on 10 qubits is compared to the tensor product of single-qubit states, and controlled gates are compared to full matrices built from projectors for every control/target pair. Grover's algorithm has tests for the oracle and the diffuser, the success probability with one or several solutions, overshooting with too many iterations, and invalid input. The QFT and inverse QFT are compared to the full DFT matrix and its inverse for 1 to 4 qubits, and tested for undoing each other, running on part of a register, qubit order, and invalid input.