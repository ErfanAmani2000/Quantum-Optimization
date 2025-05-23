# ================================================================
#  QAOA – driver–rider assignment for the 8-node instance
#  (8 binary variables  →  8 qubits  →  desktop-friendly)
# ================================================================
import cirq, qsimcirq as qsim
import numpy as np
from itertools import product, permutations
from scipy.optimize import minimize

# ----------------------------------------------------------------
# 1.  Instance data
# ----------------------------------------------------------------
BIG_M = 1_000_000
t = np.array([
    [0,     BIG_M, 10,   8,  7, 15, BIG_M, BIG_M],
    [BIG_M, 0,     13,   7,  9, 12, BIG_M, BIG_M],
    [BIG_M, BIG_M, 0,    4, BIG_M, 11, BIG_M, 10],
    [BIG_M, BIG_M, 4,    0,  6,  6, BIG_M,  9],
    [BIG_M, BIG_M, BIG_M, 6, 0,  3, 10, BIG_M],
    [BIG_M, BIG_M, 11,   6,  3, 0,  3, BIG_M],
    [BIG_M, BIG_M, BIG_M, BIG_M, BIG_M, BIG_M, 0,  8],
    [BIG_M, BIG_M, BIG_M, BIG_M, BIG_M, BIG_M, 8,  0],
], dtype=float)

n = t.shape[0]

drivers = ["D1", "D2"]
riders  = ["R1", "R2", "R3", "R4"]

origin = {"D1": 1, "D2": 2, "R1": 3, "R2": 4, "R3": 5, "R4": 6}
dest   = {"D1": 7, "D2": 8, "R1": 7, "R2": 7, "R3": 8, "R4": 8}

# ----------------------------------------------------------------
# 2.  All-pairs shortest paths  (Floyd–Warshall)
# ----------------------------------------------------------------
dist = t.copy()
for k in range(n):
    for i in range(n):
        for j in range(n):
            if dist[i, j] > dist[i, k] + dist[k, j]:
                dist[i, j] = dist[i, k] + dist[k, j]

# ----------------------------------------------------------------
# 3.  Cheapest route length  start → (pick/drop set) → end
#     capacity ≤ 2 riders  →  explicit enumeration is fine
# ----------------------------------------------------------------
def route_len(start, end, r_set):
    rs = list(r_set)
    if not rs:
        return dist[start - 1, end - 1]

    best = BIG_M

    # enumerate every precedence-feasible sequence
    if len(rs) == 1:
        (r,) = rs
        sequences = [[("p", r), ("d", r)]]
    else:                               # exactly two riders
        r1, r2 = rs
        sequences = [
            [("p", r1), ("p", r2), ("d", r1), ("d", r2)],
            [("p", r1), ("p", r2), ("d", r2), ("d", r1)],
            [("p", r2), ("p", r1), ("d", r2), ("d", r1)],
            [("p", r2), ("p", r1), ("d", r1), ("d", r2)],
            [("p", r1), ("d", r1), ("p", r2), ("d", r2)],
            [("p", r2), ("d", r2), ("p", r1), ("d", r1)],
        ]

    for seq in sequences:
        cur, cost, ok = start, 0.0, True
        for tag, r in seq:
            nxt = origin[r] if tag == "p" else dest[r]
            seg = dist[cur - 1, nxt - 1]
            if seg >= BIG_M:
                ok = False
                break
            cost += seg
            cur = nxt
        if ok:
            cost += dist[cur - 1, end - 1]
            best = min(best, cost)
    return best

# ----------------------------------------------------------------
# 4.  Pre-compute incremental costs for the QUBO
# ----------------------------------------------------------------
base_cost = {drv: route_len(origin[drv], dest[drv], []) for drv in drivers}

lin_c, quad_c = {}, {}
for drv in drivers:
    for r in riders:
        alone = route_len(origin[drv], dest[drv], [r])
        lin_c[(drv, r)] = alone - base_cost[drv]

    for i, r1 in enumerate(riders):
        for r2 in riders[i + 1:]:
            pair = route_len(origin[drv], dest[drv], [r1, r2])
            quad_c[(drv, r1, r2)] = (
                pair - base_cost[drv] - lin_c[(drv, r1)] - lin_c[(drv, r2)]
            )

# ----------------------------------------------------------------
# 5.  Binary variables  y_{drv,r}
# ----------------------------------------------------------------
var_map = {(drv, r): k for k, (drv, r) in enumerate(product(drivers, riders))}
n_qubits = len(var_map)
qubits   = cirq.LineQubit.range(n_qubits)

# ----------------------------------------------------------------
# 6.  Build the QUBO  (objective + penalties)
# ----------------------------------------------------------------
Q = {}
P = 1_000         # big penalty weight

def add(i, j, c):
    if abs(c) < 1e-12:
        return
    if i > j:
        i, j = j, i
    Q[(i, j)] = Q.get((i, j), 0.0) + c

# — objective —
for (drv, r), idx in var_map.items():
    add(idx, idx, lin_c[(drv, r)])
for (drv, r1, r2), c in quad_c.items():
    add(var_map[(drv, r1)], var_map[(drv, r2)], c)
add(0, 0, sum(base_cost.values()))        # constant shift

# — constraint 1: each rider exactly one driver —
for r in riders:
    idxs = [var_map[(drv, r)] for drv in drivers]   # two variables
    # (y_D1,r + y_D2,r − 1)²
    for idx in idxs:
        add(idx, idx,  P)
        add(idx, 0,   -2*P)
    add(0, 0, P)
    add(idxs[0], idxs[1], 2*P)

# — constraint 2: driver capacity ≤ 2  (soft   (Σ y − 2)² ) —
for drv in drivers:
    idxs = [var_map[(drv, r)] for r in riders]      # four variables
    for idx in idxs:
        add(idx, idx,   P)
        add(idx, 0,    -4*P)
    add(0, 0, 4*P)
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            add(idxs[i], idxs[j], 2*P)

# ----------------------------------------------------------------
# 7.  QUBO → Ising Hamiltonian
# ----------------------------------------------------------------
def qubo_to_hamiltonian(Q):
    H = cirq.PauliSum()
    for (i, j), c in Q.items():
        if i == j:
            H += (c / 2) * (1 - cirq.Z(qubits[i]))
        else:
            H += (c / 4) * (1 - cirq.Z(qubits[i]) * cirq.Z(qubits[j]))
    return H

H = qubo_to_hamiltonian(Q)

# ----------------------------------------------------------------
# 8.  One-layer QAOA circuit
# ----------------------------------------------------------------
def build_qaoa(gamma, beta):
    circ = cirq.Circuit()
    circ.append(cirq.H.on_each(*qubits))

    # cost e^{-i γ H}
    for term in H:
        qs = term.qubits
        c  = term.coefficient.real
        if not qs:
            continue
        if len(qs) == 1:
            (q,) = qs
            circ.append(cirq.Z(q) ** (-gamma * c / np.pi))
        else:
            q1, q2 = qs
            circ.append(cirq.ZZ(q1, q2) ** (-gamma * c / np.pi))

    # mixer e^{-i β Σ X}
    circ.append(cirq.rx(2 * beta).on_each(*qubits))
    circ.append(cirq.measure(*qubits, key="m"))
    return circ

sim = qsim.QSimSimulator()

# expectation ⟨H⟩ from full state-vector (8 qubits ⇒ fast)
def expectation(statevec):
    exp = 0.0
    for bits in product([0, 1], repeat=n_qubits):
        idx  = int("".join(map(str, bits[::-1])), 2)
        prob = abs(statevec[idx]) ** 2
        cost = sum(c * bits[i] * bits[j] for (i, j), c in Q.items())
        exp += prob * cost
    return exp

def qaoa_cost(params):
    γ, β = params
    sv = sim.simulate(build_qaoa(γ, β)).final_state_vector
    return expectation(sv)

# crude two-parameter search (COBYLA is fine for 2 vars)
opt = minimize(qaoa_cost, [0.1, 0.1], method="COBYLA")
γ_opt, β_opt = opt.x
print(f"\n🔍  Optimal (γ, β) = ({γ_opt:.4f}, {β_opt:.4f})"
      f"   ⟨cost⟩ = {opt.fun:.2f}")

# ----------------------------------------------------------------
# 9.  Sample the best circuit  &  decode assignment
# ----------------------------------------------------------------
best_circ = build_qaoa(γ_opt, β_opt)
samples   = sim.run(best_circ, repetitions=2000)
bitstring = max(samples.histogram(key="m").items(),
                key=lambda kv: kv[1])[0]
assign    = format(bitstring, f"0{n_qubits}b")[::-1]

print("\n✅  Most frequent assignment  (1 = rider assigned):")
for (drv, r), idx in var_map.items():
    print(f"  {drv}  ←  {r} : {assign[idx]}")
