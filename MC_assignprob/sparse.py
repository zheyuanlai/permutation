import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scipy.sparse import csr_matrix

# ===============================
# Utility functions for the Ising model
# ===============================
def hamiltonian(x):
    """Compute the Hamiltonian H(x) = sum_{i=1}^{d-1} (1 - x[i]*x[i+1])."""
    return np.sum(1 - x[:-1] * x[1:])

def metropolis_transition(state, beta):
    """
    Given a state vector (of spins) and inverse temperature beta,
    propose flipping a random spin and accept/reject according
    to Metropolis-Hastings.
    """
    d = len(state)
    j = np.random.randint(d)
    new_state = state.copy()
    new_state[j] *= -1
    delta_H = hamiltonian(new_state) - hamiltonian(state)
    acceptance = min(1.0, np.exp(-beta * delta_H))
    if np.random.rand() < acceptance:
        return new_state
    else:
        return state

# ===============================
# Sampling a restricted state space
# ===============================
def simulate_chain_states(d, beta, num_steps, initial_state):
    """
    Since the full state space has 2^d states (infeasible for d=50),
    we simulate a chain and store only the unique states visited.
    Returns a chain of indices and a list (states_list) containing the visited states.
    """
    state_dict = {}  # Map state (as tuple) -> index
    states_list = []  # List of visited states (as numpy arrays)
    
    def add_state(state):
        t_state = tuple(state)
        if t_state not in state_dict:
            state_dict[t_state] = len(states_list)
            states_list.append(state.copy())
    
    current_state = initial_state.copy()
    add_state(current_state)
    chain_indices = [state_dict[tuple(current_state)]]
    
    for _ in range(num_steps):
        current_state = metropolis_transition(current_state, beta)
        add_state(current_state)
        chain_indices.append(state_dict[tuple(current_state)])
        
    return np.array(chain_indices), states_list

# ===============================
# Building the sparse transition matrix
# ===============================
def build_sparse_P(states_list, beta):
    """
    For each visited state we know that only d neighbors (from one spin-flip)
    and the self-loop have nonzero transition probabilities.
    We build the transition matrix P in sparse format over the visited states.
    """
    n = len(states_list)
    d = len(states_list[0])
    rows, cols, data = [], [], []
    
    # For fast lookup of states, build a dictionary from state tuple to index.
    state_to_index = {tuple(state): i for i, state in enumerate(states_list)}
    
    for i, state in enumerate(states_list):
        total_prob = 0.0
        # Consider each single-spin flip neighbor.
        for j in range(d):
            new_state = state.copy()
            new_state[j] *= -1
            t_new = tuple(new_state)
            # Only include neighbor if it is in the visited set.
            if t_new in state_to_index:
                k = state_to_index[t_new]
                delta_H = hamiltonian(new_state) - hamiltonian(state)
                acceptance = min(1.0, np.exp(-beta * delta_H))
                prob = (1.0 / d) * acceptance
                rows.append(i)
                cols.append(k)
                data.append(prob)
                total_prob += prob
        # Self-loop to account for rejected moves.
        rows.append(i)
        cols.append(i)
        data.append(1 - total_prob)
    
    P_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
    return P_sparse

# ===============================
# Assignment problem and cost function
# ===============================
def KL_divergence(P, Q):
    """
    Compute the KL divergence D_KL^π(P || Q) using the stationary distribution.
    For demonstration we assume the stationary distribution is uniform on the visited states.
    """
    n = P.shape[0]
    pi = np.ones(n) / n  # uniform stationary distribution
    kl_div = 0.0
    for x in range(n):
        for y in range(n):
            p_val = P[x, y]
            q_val = Q[x, y]
            if p_val > 0 and q_val > 0:
                kl_div += pi[x] * p_val * np.log(p_val / q_val)
    return kl_div

def compute_cost(P, psi, gamma=0.1):
    """
    Given a permutation psi (represented as a list where psi[i] is the index that state i is mapped to),
    compute the cost as: -KL_divergence(P, P_bar) - gamma*(# of swaps). Here, P_bar = 1/2(P + QPQ),
    where Q is induced by psi.
    """
    n = P.shape[0]
    # Build permutation matrix Q explicitly from psi (sparse in nature—only one 1 per row)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, psi[i]] = 1
    P_bar = 0.5 * (P + Q @ P @ Q)
    swaps = sum(1 for i in range(n) if psi[i] != i)
    num_swaps = swaps / 2.0  # each swap counts twice
    return -KL_divergence(P, P_bar) - gamma * num_swaps

def solve_assignment(P):
    """
    Here we assume that we want to decide a swap for each pair (or leave as identity).
    For demonstration, we restrict to considering only pairs (i,j) with i <= j.
    The cost matrix is built using compute_cost for the swap that exchanges i and j (if i != j)
    or keeps state i fixed (if i == j).
    """
    n = P.shape[0]
    cost_matrix = np.zeros((n, n))
    # Build cost matrix. For each pair (i, j) consider the permutation that swaps i and j.
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
    
    # Each state i must appear in exactly one chosen pair.
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

def construct_Q(psi):
    """
    Construct the permutation matrix Q from the permutation vector psi.
    (For the projected sampler we need Q to compute QPQ.)
    """
    n = len(psi)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, psi[i]] = 1
    return Q

# ===============================
# Simulate chains using the (dense) transition matrices from the visited set.
# ===============================
def sample_next_state(current_index, P_dense):
    """Given the current state index and transition matrix P_dense (as a dense array), sample the next state index."""
    return np.random.choice(np.arange(P_dense.shape[0]), p=P_dense[current_index])

def simulate_chain(P_dense, num_steps, initial_index):
    """Simulate a Markov chain using transition matrix P_dense."""
    indices = [initial_index]
    for _ in range(num_steps):
        next_index = sample_next_state(indices[-1], P_dense)
        indices.append(next_index)
    return indices

def compute_magnetization_chain(chain, states_list):
    """
    For each visited state (indexed by chain), compute the magnetization (average spin).
    """
    mags = []
    for idx in chain:
        state = states_list[idx]
        mags.append(np.mean(state))
    return np.array(mags)

# ===============================
# Main demonstration code
# ===============================
if __name__ == '__main__':
    # Set parameters. For d=50, we cannot enumerate 2^50 states.
    d = 20
    beta = 2.5
    num_steps = 10000  # Use this many steps to build a representative visited state set.
    initial_state = np.ones(d, dtype=int)  # Start with all spins +1.

    # Simulate a chain and record the visited states.
    chain_indices, states_list = simulate_chain_states(d, beta, num_steps, initial_state)
    print("Number of visited states:", len(states_list))
    
    # Build a sparse transition matrix for the visited states.
    P_sparse = build_sparse_P(states_list, beta)
    # For assignment and simulation purposes, convert to dense.
    P_dense = P_sparse.toarray()
    print("Built P_sparse and P_dense.")
    
    # Solve the assignment problem (over the visited states) to choose an optimal permutation.
    psi_opt = solve_assignment(P_dense)
    print("Optimal permutation psi:", psi_opt)
    
    # Build Q from the optimal permutation and form the projected sampler:
    Q_opt = construct_Q(psi_opt)
    P_bar = 0.5 * (P_dense + Q_opt @ P_dense @ Q_opt)
    
    # Simulate two chains: one using the original sampler, one using the projected sampler.
    sim_steps = 100000
    initial_index = 0  # Start from the same visited state.
    chain_original = simulate_chain(P_dense, sim_steps, initial_index)
    chain_projected = simulate_chain(P_bar, sim_steps, initial_index)
    
    mag_original = compute_magnetization_chain(chain_original, states_list)
    mag_projected = compute_magnetization_chain(chain_projected, states_list)
    
    # Plot magnetization trajectories for comparison.
    plt.figure(figsize=(10, 6))
    plt.plot(mag_original, label="Original Sampler P", alpha=0.7)
    plt.plot(mag_projected, label="Projected Sampler P_bar(Q)", alpha=0.7)
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization (Average Spin)")
    plt.title("Comparison: Original vs. Projected Sampler")
    plt.legend()
    plt.grid(True)
    plt.savefig("magnetization_comparison.png")
    plt.show()
