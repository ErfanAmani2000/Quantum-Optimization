import cirq
import numpy as np
from scipy.optimize import minimize
from itertools import product

bigM = 100000
# 1. Problem Setup
t_matrix = [
    [0, 10, 20, bigM],
    [bigM, 0, 15, 25],
    [bigM, 15, 0, 5],
    [bigM, bigM, bigM, 0]
]
arcs = [(1, 2), (1, 3), (2, 3), (3, 2), (2, 4), (3, 4)]
users = ['D', 'R1', 'R2']

# 2. Map variables to indices
var_map = {}
idx = 0
for u in users:
    for arc in arcs:
        var_map[(u, arc)] = idx
        idx += 1

n_qubits = len(var_map)
qubits = [cirq.LineQubit(i) for i in range(n_qubits)]

# 3. Define QUBO cost function
Q = {}

def add_qubo(Q, i, j, val):
    if i > j:
        i, j = j, i
    Q[(i, j)] = Q.get((i, j), 0.0) + val

# 3.1 Objective term
for u in users:
    for arc in arcs:
        i, j = arc
        t_ij = t_matrix[i - 1][j - 1]
        idx = var_map[(u, arc)]
        add_qubo(Q, idx, idx, t_ij)

# 3.2 Constraints
λ = 100

def add_constraint(Q, terms, rhs, penalty=λ):
    for i, (vi, ai) in enumerate(terms):
        for j, (vj, aj) in enumerate(terms):
            add_qubo(Q, vi, vj, penalty * ai * aj)
        add_qubo(Q, vi, vi, -2 * penalty * ai * rhs)
    add_qubo(Q, 0, 0, penalty * rhs ** 2)

for u in users:
    f = lambda arc: var_map[(u, arc)]
    add_constraint(Q, [(f((1,2)),1), (f((1,3)),1)], 1)
    add_constraint(Q, [(f((2,4)),1), (f((3,4)),1)], 1)
    add_constraint(Q, [(f((1,2)),1), (f((3,2)),1), (f((2,4)),-1), (f((2,3)),-1)], 0)
    add_constraint(Q, [(f((1,3)),1), (f((2,3)),1), (f((3,4)),-1), (f((3,2)),-1)], 0)

for arc in arcs:
    for r in ['R1', 'R2']:
        i = var_map[(r, arc)]
        j = var_map[('D', arc)]
        add_qubo(Q, i, i, λ)
        add_qubo(Q, i, j, -λ)

# 4. Create cost Hamiltonian
def qubo_to_hamiltonian(Q):
    h = cirq.PauliSum()
    for (i, j), c in Q.items():
        if i == j:
            h += (c / 2) * (1 - cirq.Z(qubits[i]))
        else:
            h += (c / 4) * (1 - cirq.Z(qubits[i]) * cirq.Z(qubits[j]))
    return h

cost_ham = qubo_to_hamiltonian(Q)

# 5. Mixer Hamiltonian
def mixer_hamiltonian(qubits):
    return sum(cirq.X(q) for q in qubits)

# 6. QAOA circuit
def create_qaoa_circuit(gamma, beta, p=1):
    circuit = cirq.Circuit()
    for q in qubits:
        circuit.append(cirq.H(q))
    for _ in range(p):
        for term in cost_ham:
            coeff = term.coefficient.real
            if len(term.qubits) == 1:
                q = term.qubits[0]
                circuit.append(cirq.Z(q) ** (-gamma * coeff / np.pi))
            elif len(term.qubits) == 2:
                q1, q2 = term.qubits
                circuit.append(cirq.ZZ(q1, q2) ** (-gamma * coeff / np.pi))
        for q in qubits:
            circuit.append(cirq.rx(2 * beta)(q))
    
    # ✅ Add measurements
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit

# 7. Simulate + Expectation
sim = cirq.Simulator()

def expectation_from_statevector(statevector, Q):
    total = 0
    for bits in product([0, 1], repeat=n_qubits):
        amp = statevector[int("".join(map(str, bits[::-1])), 2)]
        prob = np.abs(amp) ** 2
        value = 0
        for (i, j), c in Q.items():
            value += c * bits[i] * bits[j]
        total += prob * value
    return total

def qaoa_expectation(params):
    gamma, beta = params
    circuit = create_qaoa_circuit(gamma, beta)
    result = sim.simulate(circuit)
    return expectation_from_statevector(result.final_state_vector, Q)

# 8. Optimize QAOA parameters
res = minimize(qaoa_expectation, x0=[0.1, 0.1], method='COBYLA')
print("\n🔍 Optimal gamma, beta:", res.x)
print("🎯 Minimum expected cost:", res.fun)

# 9. Sample best solution
best_circuit = create_qaoa_circuit(*res.x)
result = sim.run(best_circuit, repetitions=100)
hist = result.histogram(key="result")
most_common = max(hist.items(), key=lambda x: x[1])[0]
sol_bits = format(most_common, f'0{n_qubits}b')[::-1]

print("\n✅ Most frequent bitstring:")
for (key, idx) in var_map.items():
    print(f"x^{key[0]}_{key[1]} = {sol_bits[idx]}")
