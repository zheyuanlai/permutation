import numpy as np
import matplotlib.pyplot as plt

d = 50                 # Number of spins
n_steps = 10000        # Number of MCMC steps
beta = 1.0             # Inverse temperature

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

# Smoothing the data

def moving_average(data, window_size=50):
    """
    Compute the moving average of 'data' with a specified window_size.
    """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window_size = 50
smooth_mag_std = moving_average(mag_std, window_size)
smooth_mag_mod = moving_average(mag_mod, window_size)
smooth_energy_std = moving_average(energy_std, window_size)
smooth_energy_mod = moving_average(energy_mod, window_size)

smoothed_steps = np.arange(len(smooth_mag_std))

plt.figure(figsize=(8, 5))
plt.plot(smoothed_steps, smooth_mag_std, label='Standard MH (smoothed)')
plt.plot(smoothed_steps, smooth_mag_mod, label='Projection Sampler (smoothed)')
plt.xlabel('MCMC Step')
plt.ylabel('Magnetization')
plt.title('Magnetization vs. Time (Smoothed)')
plt.legend()
plt.tight_layout()
plt.savefig('ising_model_magnetization_smoothed.png')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(smoothed_steps, smooth_energy_std, label='Standard MH (smoothed)')
plt.plot(smoothed_steps, smooth_energy_mod, label='Projection Sampler (smoothed)')
plt.xlabel('MCMC Step')
plt.ylabel('Hamiltonian H(x)')
plt.title('Energy vs. Time (Smoothed)')
plt.legend()
plt.tight_layout()
plt.savefig('ising_model_energy_smoothed.png')
plt.show()
