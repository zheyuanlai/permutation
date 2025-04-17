import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scipy.sparse import csr_matrix

# ===============================
# Utility functions for the Ising model
# ===============================
def hamiltonian(x):
    """Compute H(x)=∑_{k=1}^{d-1}(1 - x[k]*x[k+1])."""
    return np.sum(1 - x[:-1] * x[1:])

def metropolis_transition(state, beta):
    """One Metropolis step: flip a random spin."""
    d = len(state)
    j = np.random.randint(d)
    new_state = state.copy()
    new_state[j] *= -1
    ΔH = hamiltonian(new_state) - hamiltonian(state)
    if np.random.rand() < np.exp(-beta * ΔH):
        return new_state
    return state

# ===============================
# Sample a restricted state‑space by running the MH chain
# ===============================
def simulate_chain_states(d, beta, num_steps, initial_state):
    state_to_index = {}
    states = []
    def add(s):
        key = tuple(s)
        if key not in state_to_index:
            state_to_index[key] = len(states)
            states.append(s.copy())
    s = initial_state.copy()
    add(s)
    for _ in range(num_steps):
        s = metropolis_transition(s, beta)
        add(s)
    return states, state_to_index

# ===============================
# Build sparse P over visited states
# ===============================
def build_sparse_P(states, beta):
    n = len(states)
    d = len(states[0])
    rows, cols, data = [], [], []
    idx = {tuple(s): i for i, s in enumerate(states)}
    for i, s in enumerate(states):
        tot = 0.0
        for k in range(d):
            ns = s.copy()
            ns[k] *= -1
            key = tuple(ns)
            if key in idx:
                j = idx[key]
                p = (1/d) * min(1, np.exp(-beta*(hamiltonian(ns)-hamiltonian(s))))
                rows.append(i); cols.append(j); data.append(p)
                tot += p
        # self-loop
        rows.append(i); cols.append(i); data.append(1-tot)
    return csr_matrix((data, (rows, cols)), shape=(n,n))

# ===============================
# Vectorized KL divergence
# ===============================
def kl_divergence(P, Q):
    n = P.shape[0]
    pi = np.ones(n)/n
    xs, ys = P.nonzero()
    valid = Q[xs, ys] > 0
    vals = pi[xs[valid]] * P[xs[valid], ys[valid]] * np.log(
        P[xs[valid], ys[valid]] / Q[xs[valid], ys[valid]]
    )
    return vals.sum()

# ===============================
# Solve assignment with equi‑energy restriction
# ===============================
def solve_assignment(P, energies):
    n = P.shape[0]
    # allow swaps only if energies match or i==j
    allowed = [(i,j) for i in range(n) for j in range(i,n)
               if i==j or energies[i]==energies[j]]
    print("Allowed swaps:", len(allowed))
    cost = {}
    for i,j in allowed:
        psi = list(range(n))
        if i!=j:
            psi[i],psi[j] = psi[j],psi[i]
        Pp = P[psi,:][:,psi]
        cost[(i,j)] = -kl_divergence(P, 0.5*(P+Pp))
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = { (i,j): solver.BoolVar(f"x_{i}_{j}") for (i,j) in allowed }
    # each state appears exactly once
    for i in range(n):
        solver.Add(solver.Sum(x[p] for p in allowed if i in p) == 1)
    # objective
    solver.Minimize(solver.Sum(cost[p]*x[p] for p in allowed))
    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution")
    psi = list(range(n))
    for i,j in allowed:
        if i!=j and x[(i,j)].solution_value()>0.5:
            psi[i],psi[j] = j,i
    return psi

# ===============================
# Full‑chain simulation switching to P̄ on subset
# ===============================
def simulate_full_chain(d, beta, Pbar, states, state_to_index, num_steps, initial_state):
    chain = []
    s = initial_state.copy()
    for _ in range(num_steps):
        key = tuple(s)
        if Pbar is not None and key in state_to_index:
            idx = state_to_index[key]
            probs = Pbar[idx]
            j = np.random.choice(len(states), p=probs)
            s = states[j].copy()
        else:
            s = metropolis_transition(s, beta)
        chain.append(s.copy())
    return np.array(chain)

if __name__ == '__main__':
    d, beta = 25, 2.5
    sub_steps, full_steps = 20000, 100000
    init = np.ones(d, dtype=int)

    # 1) Build subset
    states, idx_map = simulate_chain_states(d, beta, sub_steps, init)
    print("Visited subset size:", len(states))
    energies = np.array([hamiltonian(s) for s in states])

    # 2) Build P and solve assignment
    Psp = build_sparse_P(states, beta)
    P = Psp.toarray()
    psi = solve_assignment(P, energies)
    print("Subset permutation psi computed.")

    # 3) Build projected P̄
    Pp = P[psi,:][:,psi]
    Pbar = 0.5*(P + Pp)

    # 4) Simulate full chains
    orig_chain = simulate_full_chain(d, beta, None, states, idx_map, full_steps, init)
    proj_chain = simulate_full_chain(d, beta, Pbar, states, idx_map, full_steps, init)

    # 5) Compute magnetization and plot
    mag_orig = orig_chain.mean(axis=1)
    mag_proj = proj_chain.mean(axis=1)

    plt.figure(figsize=(10,6))
    plt.plot(mag_orig, label="Original MH")
    plt.plot(mag_proj, label="Projected MH")
    plt.xlabel("Step")
    plt.ylabel("Magnetization (Average Spin)")
    plt.legend()
    plt.title("Comparison: Original vs. Projected Sampler")
    plt.grid(True)
    plt.savefig("magnetization_comparison.png")
    plt.show()
