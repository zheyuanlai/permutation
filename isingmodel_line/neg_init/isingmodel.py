import numpy as np
import matplotlib.pyplot as plt

d = 50                 # Number of spins
n_steps = 100000       # Number of MCMC steps
beta = 2.0             # Inverse temperature

np.random.seed(42)

def hamiltonian(x):
    """Compute the Hamiltonian H(x) = sum_{i=1}^{d-1} (1 - x_i x_{i+1})."""
    return np.sum(1 - x[:-1] * x[1:])

def metropolis_step(x):
    """One step of standard Metropolis-Hastings for the Ising model.
       Propose flipping one random spin and accept with probability min(1, exp(-ΔH))."""
    x_new = x.copy()
    i = np.random.randint(0, d)
    x_new[i] *= -1
    delta_H = hamiltonian(x_new) - hamiltonian(x)
    if np.random.rand() < np.exp(-beta * delta_H):
        return x_new
    else:
        return x

def Q_permutation(x):
    """Permutation Q:
       If x is all +1 or all -1, leave unchanged.
       Otherwise, flip all spins: Q(x) = -x."""
    if np.all(x == 1) or np.all(x == -1):
        return x.copy()
    else:
        return -x.copy()

def metropolis_step_modified(x):
    """
    Projection sampler P(Q) = 1/2 (P + Q P Q).
    We implement by setting:
       y1 = P(x)
       y2 = Q(P(Q(x)))
    and then choose between y1 and y2 with equal probability.
    """
    y1 = metropolis_step(x)
    y2 = Q_permutation(metropolis_step(Q_permutation(x)))
    if np.random.rand() < 0.5:
        return y1
    else:
        return y2

def magnetization(x):
    """Compute the magnetization (average spin) of state x."""
    return np.mean(x)

state_std = np.array([-1] * d)
state_mod = np.array([-1] * d)

mag_std = np.zeros(n_steps)
mag_mod = np.zeros(n_steps)
energy_std = np.zeros(n_steps)
energy_mod = np.zeros(n_steps)

for t in range(n_steps):
    state_std = metropolis_step(state_std)
    state_mod = metropolis_step_modified(state_mod)
    
    mag_std[t] = magnetization(state_std)
    mag_mod[t] = magnetization(state_mod)
    energy_std[t] = hamiltonian(state_std)
    energy_mod[t] = hamiltonian(state_mod)

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

plt.figure(figsize=(8, 5))
plt.plot(energy_std, label='Standard MH')
plt.plot(energy_mod, label='Projection Sampler')
plt.xlabel('MCMC Step')
plt.ylabel('Hamiltonian H(x)')
plt.title('Energy vs. Time')
plt.legend()
plt.tight_layout()
plt.savefig('ising_model_energy.png')
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