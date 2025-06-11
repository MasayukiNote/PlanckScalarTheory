import numpy as np

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
alpha = 1/137.035999084
epsilon = 0.06
gamma = 1/3
c = 2.99792458e8
G = 6.67430e-11
Lambda_eff = 1 / (2 * np.pi * l_planck)
xi = 1e-4
Nx, Ny, Nz, Nt = 64, 64, 64, 256

Phi = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6)) + \
      1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6))

def compute_Tmunu(Phi, k):
    delta_Phi = epsilon * np.exp(1j * k * l_planck) * np.ones_like(Phi)
    grad_mu = np.zeros_like(Phi)
    for mu in range(4):
        grad_mu += (np.roll(delta_Phi, -1, axis=mu) - delta_Phi) / l_planck
    Tmunu = np.abs(grad_mu)**2 / (np.exp(k * l_planck) - 1)**gamma
    return Tmunu

def compute_h_plus(k, r=1.23e25, f=100):
    Phi_norm = Lambda_eff**4 / alpha**2
    G_eff = 1 / (16 * np.pi * xi * Phi_norm)
    h_plus = (16 * np.pi * G_eff * epsilon**2 * k**2) / (r * c**4 * (np.exp(k * l_planck) - 1)**gamma)
    return h_plus

k = 2 * np.pi * f / c
h_plus = compute_h_plus(k)
print(f"Gravitational wave strain: {h_plus:.3e}, Error: {abs(h_plus - 3.24e-21)/3.24e-21*100:.2f}%")
np.save("SimulationCodes/paper3/data/gw_h_plus.npy", np.array(h_plus))