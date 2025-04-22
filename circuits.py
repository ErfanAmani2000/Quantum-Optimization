from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import numpy as np

# Problem size
n_qubits = 18
p = 3  # QAOA depth
gamma = 1.26172092
beta = -0.09708315

# Create circuit
qc = QuantumCircuit(n_qubits)

# Initial layer of Hadamard gates
qc.h(range(n_qubits))

# Define sample ZZ connections based on arcs used in your Cirq code
zz_pairs = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)
]

# Apply QAOA layers
for _ in range(p):
    # Cost layer (ZZ gates)
    for i, j in zz_pairs:
        qc.cx(i, j)
        qc.rz(2 * gamma, j)
        qc.cx(i, j)
    
    # Mixer layer (Rx gates)
    for q in range(n_qubits):
        qc.rx(2 * beta, q)

# Measurement
qc.measure_all()

# Draw the circuit
qc.draw('mpl')
qc.draw('mpl', filename='qaoa_circuit.png')
plt.show()
