# lattice_simulation.py
import numpy as np
import random
from scipy import integrate
import multiprocessing as mp
from functools import partial
import lattice_cy

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
m_e = 0.51099895000e-3
Nx, Ny, Nz, Nt = 48, 48, 48, 96
N_fields = 8  # Updated to 8 components
a_t = 1.0

Phi = np.random.normal(0, hbar_c/l_planck, (Nx, Ny, Nz, Nt, N_fields)) + \
      1j * np.random.normal(0, hbar_c/l_planck, (Nx, Ny, Nz, Nt, N_fields))

def metropolis_step(Phi, step, include_spin=False, include_constraint=False):
    Phi_new = Phi.copy()
    idx = (random.randint(0, Nx-1), random.randint(0, Ny-1),
           random.randint(0, Nz-1), random.randint(0, Nt-1),
           random.randint(0, N_fields-1))
    Phi_new[idx] += np.random.normal(0, 0.01 * hbar_c/l_planck) + \
                    1j * np.random.normal(0, 0.01 * hbar_c/l_planck)
    S_old = lattice_cy.compute_action(Phi, include_spin, include_constraint, a_t)
    S_new = lattice_cy.compute_action(Phi_new, include_spin, include_constraint, a_t)
    if S_new <= S_old or np.random.random() < np.exp(-(S_new - S_old)):
        return Phi_new, S_new
    return Phi, S_old

def run_simulation(steps=2000):
    action_history = []
    lorentz_ratios = []
    Phi_current = Phi.copy()
    with mp.Pool(8) as pool:
        for step in range(steps):
            include_constraint = (step > 1000)
            include_spin = (step > 1000)
            Phi_current, S = metropolis_step(Phi_current, step, include_spin,
                                             include_constraint)
            action_history.append(S)
            if step % 100 == 0:
                ratio = lattice_cy.compute_lorentz_ratio(Phi_current)
                lorentz_ratios.append(ratio)
                print(f"Step {step}, Action: {S:.2e}, Lorentz Ratio: {ratio:.4f}")
    np.save("action_history.npy", action_history)
    np.save("lorentz_ratios.npy", lorentz_ratios)
    np.save("Phi_final.npy", Phi_current)

if __name__ == "__main__":
    run_simulation()