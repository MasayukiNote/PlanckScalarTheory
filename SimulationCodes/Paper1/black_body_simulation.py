# black_body_simulation.py
import numpy as np
from scipy import integrate

l_planck = 1.616229e-35
h = 6.62607015e-34
c = 2.99792458e8
k_B = 1.380649e-23
alpha = 1 / 137.035999084
beta = 2.19e-7
hbar = h / (2 * np.pi)
Nx, Ny, Nz, Nt = 48, 48, 48, 96
T_values = [3000, 6000, 10000]

S_z = hbar * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
S_x = hbar / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128)
S_y = hbar / np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128)

def V_Phi_gamma(Phi, T):
    Phi_norm = np.sqrt(np.sum(np.abs(Phi[...,:3])**2, axis=-1))
    exponent = Phi_norm * l_planck * (h * c / (k_B * T))
    return alpha / l_planck**2 * np.sum(Phi_norm / (np.exp(exponent) - 1))

def pseudo_spin_term(Phi):
    Phi_gamma = Phi[...,:3]
    spin = np.einsum('...i,ij,...j->...', np.conj(Phi_gamma), S_z, Phi_gamma).real
    return beta * np.sum(spin)

def compute_action_black_body(Phi, T):
    kinetic = np.sum(np.abs(np.roll(Phi, -1, axis=(0,1,2,3)) - Phi)**2) / l_planck**2
    potential = V_Phi_gamma(Phi, T)
    spin = pseudo_spin_term(Phi)
    return (kinetic - potential + spin) * l_planck**4

def u_nu(nu, T):
    return (8 * np.pi * nu**2 / c**3) * (h * nu / (np.exp(h * nu / (k_B * T)) - 1))

u_theoretical = []
for T in T_values:
    u_total, _ = integrate.quad(lambda nu: u_nu(nu, T), 0, 1e20)
    u_theoretical.append(u_total)

Phi_gamma = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 3)) + \
            1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 3))

def metropolis_step_black_body(Phi, T, step):
    Phi_new = Phi.copy()
    idx = (np.random.randint(0, Nx), np.random.randint(0, Ny),
           np.random.randint(0, Nz), np.random.randint(0, Nt),
           np.random.randint(0, 3))
    Phi_new[idx] += np.random.normal(0, 0.01 * l_planck) + \
                    1j * np.random.normal(0, 0.01 * l_planck)
    S_old = compute_action_black_body(Phi, T)
    S_new = compute_action_black_body(Phi_new, T)
    if S_new <= S_old or np.random.random() < np.exp(-(S_new - S_old)):
        return Phi_new, S_new
    return Phi, S_old

u_simulated = []
for T in T_values:
    Phi_current = Phi_gamma.copy()
    for step in range(2000):
        Phi_current, S = metropolis_step_black_body(Phi_current, T, step)
    Phi_norm = np.mean(np.sqrt(np.sum(np.abs(Phi_current)**2, axis=-1)))
    exponent = Phi_norm * l_planck * (h * c / (k_B * T))
    u = (8 * np.pi**5 * (k_B * T)**4 / (15 * (h * c)**3)) / (np.exp(exponent) - 1)
    u_simulated.append(u)
    np.save(f"Phi_black_body_T{T}.npy", Phi_current)