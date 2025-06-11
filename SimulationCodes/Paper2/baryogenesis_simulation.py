import numpy as np
from common_utils import metropolis_step

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
alpha = 1/137.035999084
kappa = 1e-10
Nx, Ny, Nz, Nt = 64, 64, 64, 256

Phi = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 8)) + \
      1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 8))

def compute_action_baryogenesis(Phi):
    kinetic = np.sum(np.abs(np.roll(Phi, -1, axis=(0,1,2,3)) - Phi)**2) / l_planck**2
    baryon = kappa * np.sum(np.abs(Phi[...,4:])**2) / l_planck**2
    return (kinetic + baryon) * l_planck**4

def run_baryogenesis_simulation():
    global Phi
    for step in range(5000):
        Phi, S = metropolis_step(Phi, compute_action_baryogenesis, 0.01)
        if step % 100 == 0:
            print(f"Step {step}, Action: {S:.2e}")
    n_B = kappa * np.mean(np.abs(Phi[...,4:])**2) / l_planck**3
    print(f"Baryon asymmetry: {n_B:.3e}")
    np.save("SimulationCodes/paper2/data/Phi_baryogenesis.npy", Phi)
    np.save("SimulationCodes/paper2/data/baryon_asymmetry.npy", np.array(n_B))

if __name__ == "__main__":
    run_baryogenesis_simulation()