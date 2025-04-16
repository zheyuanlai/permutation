import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import itertools

def hamiltonian(x):
    """Compute the Hamiltonian: H(x) = sum_{i=1}^{d-1} (1 - x[i] * x[i+1])."""
    return np.sum(1 - x[:-1] * x[1:])

def stationary_distribution_reversible(P):
    """Compute the stationary distribution for a reversible transition matrix P."""
    n = P.shape[0]
    pi_unnormalized = np.ones(n)
    for i in range(1, n):
        pi_unnormalized[i] = pi_unnormalized[i-1] * (P[i-1, i] / P[i, i-1])
    pi = pi_unnormalized / np.sum(pi_unnormalized)
    return pi

def KL_divergence(P, Q):
    """Compute the KL divergence D_KL^pi(P || Q) using the stationary distribution from P."""
    kl_div = 0.0
    n = P.shape[0]
    pi = stationary_distribution_reversible(P)
    for x in range(n):
        for y in range(n):
            if P[x, y] > 0 and Q[x, y] > 0:
                kl_div += pi[x] * P[x, y] * np.log(P[x, y] / Q[x, y])
    return kl_div    

# -----------------------------
# Construct the state space for the Ising model on a line with d spins.
# -----------------------------
d = 5  # number of spins (for demonstration)
# Each state is a vector of d spins, each either +1 or -1.
state_space = []
for bits in itertools.product([-1, 1], repeat=d):
    state_space.append(np.array(bits))
num_states = len(state_space)

# -----------------------------
# Build transition matrix P.
# -----------------------------
beta = 1.5
P = np.zeros((num_states, num_states))
proposal_prob = 1.0 / d

def find_state_index(state_space, x):
    for idx, state in enumerate(state_space):
        if np.array_equal(state, x):
            return idx
    raise ValueError("State not found in state_space.")

for i, x in enumerate(state_space):
    total = 0.0
    for j in range(d):
        y = x.copy()
        y[j] *= -1
        k = find_state_index(state_space, y)
        delta_H = hamiltonian(y) - hamiltonian(x)
        acceptance = min(1.0, np.exp(-beta * delta_H))
        P[i, k] = proposal_prob * acceptance
        total += P[i, k]
    P[i, i] = 1 - total

def compute_cost(P, psi, gamma=0.1):
    n = P.shape[0]
    Q = np.zeros_like(P)
    for i in range(n):
        Q[i, psi[i]] = 1
    P_bar = 0.5 * (P + Q @ P @ Q)
    swaps = sum(1 for i in range(n) if psi[i] != i)
    num_swaps = swaps / 2.0
    return -KL_divergence(P, P_bar) - gamma * num_swaps

def solve_assignment(P):
    n = P.shape[0]
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            psi = list(range(n))
            if i != j:
                psi[i], psi[j] = psi[j], psi[i]
            cost_matrix[i, j] = compute_cost(P, psi)
            cost_matrix[j, i] = cost_matrix[i, j]
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if solver is None:
        print("Failed to create solver.")
        return None

    x_vars = {}
    for i in range(n):
        for j in range(i, n):
            x_vars[(i, j)] = solver.BoolVar(f'x_{i}_{j}')
    
    for i in range(n):
        terms = []
        for j in range(i, n):
            terms.append(x_vars[(i, j)])
        for j in range(0, i):
            terms.append(x_vars[(j, i)])
        solver.Add(solver.Sum(terms) == 1)
    
    objective_terms = []
    for i in range(n):
        for j in range(i, n):
            objective_terms.append(cost_matrix[i, j] * x_vars[(i, j)])
    solver.Minimize(solver.Sum(objective_terms))
    
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("Solver did not find an optimal solution.")
        return None
    
    psi = [None] * n
    for i in range(n):
        for j in range(i, n):
            if x_vars[(i, j)].solution_value() > 0.5:
                if i == j:
                    psi[i] = i
                else:
                    psi[i] = j
                    psi[j] = i
    return psi

psi_opt = solve_assignment(P)
print("Optimal permutation psi:", psi_opt)


def sample_next_state(current_index, P):
    """Given the current state index and transition matrix P, sample the next state index."""
    return np.random.choice(np.arange(P.shape[0]), p=P[current_index])

def simulate_chain(P, num_steps, initial_index):
    """Simulate a chain from the transition matrix P for a given number of steps."""
    indices = [initial_index]
    for _ in range(num_steps):
        next_index = sample_next_state(indices[-1], P)
        indices.append(next_index)
    return indices

def compute_magnetization_chain(chain, state_space):
    """Compute the magnetization (average spin) along the chain."""
    mags = []
    for idx in chain:
        state = state_space[idx]
        mags.append(np.mean(state))
    return np.array(mags)

def construct_Q(psi):
    n = len(psi)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, psi[i]] = 1
    return Q

Q_opt = construct_Q(psi_opt)

P_bar = 0.5 * (P + Q_opt @ P @ Q_opt)

# -----------------------------
# Simulate chains and plot the performance
# -----------------------------
num_steps = 1000
initial_index = 0

chain_original = simulate_chain(P, num_steps, initial_index)
chain_projected = simulate_chain(P_bar, num_steps, initial_index)

mag_original = compute_magnetization_chain(chain_original, state_space)
mag_projected = compute_magnetization_chain(chain_projected, state_space)

plt.figure(figsize=(10, 6))
plt.plot(mag_original, label="Original Sampler P", alpha=0.7)
plt.plot(mag_projected, label="Projected Sampler P_bar(Q)", alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("Magnetization (Average Spin)")
plt.title("Performance Comparison: Original vs. Projected Sampler")
plt.legend()
plt.grid(True)
plt.show()