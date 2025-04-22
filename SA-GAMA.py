import numpy as np
import random
import itertools
import matplotlib.pyplot as plt

# Problem setup
bigM = 100000
t_matrix = np.array([
    [0, 10, 20, bigM],     # from 1
    [bigM, 0, 15, 25],     # from 2
    [bigM, 15, 0, 5],      # from 3
    [bigM, bigM, bigM, 0]  # from 4
])

agents = ['D', 'R1', 'R2']
edges = [(1, 2), (1, 3), (2, 3), (3, 2), (2, 4), (3, 4)]

# Variable indexing
variable_index = {}
index_variable = {}
idx = 0
for u in agents:
    for (i, j) in edges:
        variable_index[(u, i, j)] = idx
        index_variable[idx] = (u, i, j)
        idx += 1
n_vars = len(variable_index)

# Build QUBO
Q = np.zeros((n_vars, n_vars))
penalty = 5000

# Objective: travel cost
for (u, i, j), var in variable_index.items():
    Q[var, var] += t_matrix[i-1][j-1]

# Constraint helper
def add_exactly_one(Q, indices, penalty):
    for i in indices:
        Q[i, i] += penalty
        for j in indices:
            if i != j:
                Q[i, j] += 2 * penalty
        Q[i, i] -= 2 * penalty
    return Q

# Entry and exit constraints
for u in agents:
    Q = add_exactly_one(Q, [variable_index[(u, 1, 2)], variable_index[(u, 1, 3)]], penalty)
    Q = add_exactly_one(Q, [variable_index[(u, 2, 4)], variable_index[(u, 3, 4)]], penalty)

# D subset of R1 and R2
for (i, j) in edges:
    d = variable_index[('D', i, j)]
    for r in ['R1', 'R2']:
        r_idx = variable_index[(r, i, j)]
        Q[d, d] += penalty
        Q[d, r_idx] -= penalty
        Q[r_idx, d] -= penalty

# Simulated Annealing
def simulated_annealing(Q, n_reads=3000, T_start=100.0, alpha=0.99):
    n = Q.shape[0]
    best_solution = None
    best_energy = float('inf')
    solution = np.random.randint(0, 2, n)

    def energy(x): return x @ Q @ x
    T = T_start

    for _ in range(n_reads):
        i = np.random.randint(n)
        candidate = solution.copy()
        candidate[i] ^= 1  # flip bit
        delta = energy(candidate) - energy(solution)

        if delta < 0 or random.random() < np.exp(-delta / T):
            solution = candidate
            e = energy(solution)
            if e < best_energy:
                best_solution = candidate.copy()
                best_energy = e
        T *= alpha

    return best_solution, best_energy

# Graver-like augmentation
def augment(solution, Q):
    n = len(solution)
    best = solution.copy()
    best_energy = best @ Q @ best
    improved = True

    while improved:
        improved = False
        for i, j in itertools.combinations(range(n), 2):
            candidate = best.copy()
            candidate[i] ^= 1
            candidate[j] ^= 1
            energy = candidate @ Q @ candidate
            if energy < best_energy:
                best = candidate
                best_energy = energy
                improved = True
    return best, best_energy

# GAMA with energy tracking
def gama_with_tracking(Q, seeds=10):
    energies = []
    best = None
    best_energy = float('inf')

    for _ in range(seeds):
        initial, _ = simulated_annealing(Q)
        improved, energy = augment(initial, Q)
        energies.append(energy)
        if energy < best_energy:
            best = improved
            best_energy = energy

    return best, best_energy, energies

# Run GAMA
solution, energy, energy_trace = gama_with_tracking(Q, seeds=10)

# Decode solution
paths = {u: [] for u in agents}
for idx, val in enumerate(solution):
    if val == 1:
        u, i, j = index_variable[idx]
        paths[u].append((i, j))

# Print solution
print("\nBest GAMA Solution (Energy = {:.2f}):".format(energy))
for u in agents:
    print(f"  {u}: {paths[u]}")

# Plot energy per seed
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(energy_trace)+1), energy_trace, marker='o')
plt.title("Energy per Seed in GAMA")
plt.xlabel("Seed Number")
plt.ylabel("Final Energy after Augmentation")
plt.grid(True)
plt.tight_layout()
plt.show()
