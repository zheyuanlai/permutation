import numpy as np
import matplotlib.pyplot as plt

d = 50                 # Number of spins
n_steps = 20000        # Number of MCMC steps
beta = 2.0             # Inverse temperature

np.random.seed(42)

def generate_J_discrete_SK(d, scale=True, seed=None):
    """
    Generate symmetric coupling matrix J for the SK model with discrete ±1 couplings.
    J_ij ∈ {-1, +1}, J_ii = 0, and J is symmetric.
    If scale=True, scales by 1/sqrt(d) to match standard SK normalization.
    """
    if seed is not None:
        np.random.seed(seed)

    J = np.random.choice([-1, 1], size=(d, d))
    J = np.triu(J, 1)
    J = J + J.T
    np.fill_diagonal(J, 0)

    if scale:
        J = J / np.sqrt(d)

    return J

J = generate_J_discrete_SK(d)

def hamiltonian(x, J):
    """
    Compute the SK Hamiltonian: H(x) = - sum_{i < j} J_ij x_i x_j
    """
    return -0.5 * np.sum(J * np.outer(x, x))

def metropolis_step(x):
    """One step of standard Metropolis-Hastings for the Ising model.
       Propose flipping one random spin and accept with probability min(1, exp(-ΔH))."""
    x_new = x.copy()
    i = np.random.randint(0, d)
    x_new[i] *= -1
    delta_H = hamiltonian(x_new,J) - hamiltonian(x,J)
    if np.random.rand() < np.exp(-beta * delta_H):
        return x_new
    else:
        return x

# Apply permutation Q_psi
def Q_psi(x, psi):
    key = x.tobytes()
    return psi[key].copy() if key in psi else x.copy()

# Efficient update of the adaptive permutation psi
def update_psi_efficient(psi, history, energy_history, update_interval=100):
    if len(history) % update_interval == 0:
        # Select a random pair of states with matching energy to update the permutation
        energy_to_states = {}
        for i, energy in enumerate(energy_history):
            if energy not in energy_to_states:
                energy_to_states[energy] = []
            energy_to_states[energy].append(history[i])

        # Find matching energy states and update permutation if necessary
        for energy, states in energy_to_states.items():
            if len(states) > 1:
                i, j = np.random.choice(len(states), 2, replace=False)
                x_i = states[i]
                x_j = states[j]
                xi_key, xj_key = x_i.tobytes(), x_j.tobytes()

                if xi_key not in psi and xj_key not in psi:
                    psi[xi_key] = x_j
                    psi[xj_key] = x_i
                    break

# Adaptive projection step
def metropolis_step_adaptive_projection(x, psi):
    y1 = metropolis_step(x)
    y2 = Q_psi(metropolis_step(Q_psi(x, psi)), psi)
    return y1 if np.random.rand() < 0.5 else y2

# Magnetization
def magnetization(x):
    return np.mean(x)

# Initialization
state_std = np.ones(d, dtype=int)
state_mod = state_std.copy()
mag_std = np.zeros(n_steps)
mag_mod = np.zeros(n_steps)
energy_std = np.zeros(n_steps)
energy_mod = np.zeros(n_steps)
psi = {}
history = []
energy_history = []

# Run both standard MH and adaptive sampler
for t in range(n_steps):
    state_std = metropolis_step(state_std)  # Standard MH step
    state_mod = metropolis_step_adaptive_projection(state_mod, psi)  # Adaptive step

    mag_std[t] = magnetization(state_std)
    mag_mod[t] = magnetization(state_mod)
    energy_std[t] = hamiltonian(state_std,J)
    energy_mod[t] = hamiltonian(state_mod,J)

    # Record history and update psi for adaptive sampler
    history.append(state_mod.copy())
    energy_history.append(energy_mod[t])
    update_psi_efficient(psi, history, energy_history)

# Plot results
plt.figure(figsize=(14, 6))

# Magnetization plot
plt.subplot(1, 2, 1)
plt.plot(mag_std, label='Standard MH')
plt.plot(mag_mod, label='Adaptive Projection Sampler')
plt.xlabel('MCMC Step')
plt.ylabel('Magnetization')
plt.title('Magnetization vs. Time')
plt.legend()

# Hamiltonian plot
plt.subplot(1, 2, 2)
plt.plot(energy_std, label='Standard MH')
plt.plot(energy_mod, label='Adaptive Projection Sampler')
plt.xlabel('MCMC Step')
plt.ylabel('Hamiltonian H(x)')
plt.title('Energy vs. Time')
plt.legend()

plt.tight_layout()
plt.savefig('ising_model_comparison_optimized.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(mag_std, label='Standard MH')
plt.plot(mag_mod, label='Projection Sampler')
plt.xlabel('MCMC Step')
plt.ylabel('Magnetization')
plt.title('Magnetization vs. Time')
plt.legend()
plt.tight_layout()
plt.savefig('ising_model_magnetization.png')
plt.show()

def autocorr(x, max_lag=1000):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    result = result[result.size // 2:]
    return result[:max_lag] / result[0]

max_lag = 1000
ac_std = autocorr(mag_std, max_lag)
ac_mod = autocorr(mag_mod, max_lag)

plt.figure(figsize=(8, 5))
plt.plot(ac_std, label='Standard MH')
plt.plot(ac_mod, label='Projection Sampler')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Magnetization')
plt.legend()
plt.tight_layout()
plt.savefig('ising_autocorrelation.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(mag_std[-100000:], bins=30, alpha=0.5, label='Standard MH $P$', density=True)
plt.hist(mag_mod[-100000:], bins=30, alpha=0.5, label='Projection Sampler $\dfrac{1}{2}(P+QPQ)$', density=True)
plt.xlabel('Magnetization')
plt.ylabel('Density')
plt.title('Histogram of Magnetization (last 5000 steps)')
plt.legend()
plt.tight_layout()
plt.savefig('ising_histogram.png')
plt.show()