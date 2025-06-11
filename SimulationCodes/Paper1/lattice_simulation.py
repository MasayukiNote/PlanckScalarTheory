import numpy as np
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))
from lattice_cy import compute_action

# Constants
l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
c = 2.99792458e8
alpha = 1/137.035999084
mu_sq = -0.5
lambda_ = 1e-73
epsilon = 0.06
gamma = 1/3
beta = 2.19e-5
kappa_vac = 2.61e-122
xi = 1e-4
Nx, Ny, Nz, Nt = 64, 64, 64, 256
N_sweeps = 5000

# Initialize field
Phi = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6)) + \
      1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6))

def metropolis_step(phi, action_func, step_size=0.01):
    phi_new = phi.copy()
    idx = (
        np.random.randint(0, Nx), np.random.randint(0, Ny),
        np.random.randint(0, Nz), np.random.randint(0, Nt),
        np.random.randint(0, 6)
    )
    phi_new[idx] += np.random.normal(0, step_size * l_planck) + \
                    1j * np.random.normal(0, step_size * l_planck)
    S_old = action_func(phi)
    S_new = action_func(phi_new)
    if S_new <= S_old or np.random.random() < np.exp(-(S_new - S_old)):
        return phi_new, S_new
    return phi, S_old

def run_simulation(output_path="SimulationCodes/paper1/data/Phi_final.npy"):
    global Phi
    action_history = []
    start_time = time.time()
    for step in range(N_sweeps):
        Phi, action = metropolis_step(Phi, lambda p: compute_action(
            l_planck, mu_sq, lambda_, epsilon, alpha, gamma, beta,
            kappa_vac, xi, Nx, Ny, Nz, Nt, p, paper_type="paper1"
        ), 0.01)
        action_history.append(action)
        if step % 100 == 0:
            print(f"Step {step}, Action: {action:.2e}, Time: {time.time() - start:.2f} s")
    np.save(output_path, Phi)
    np.save("SimulationCodes/paper1/data/action_history.npy", np.array(action_history))
    print(f"Simulation completed. Results saved to {output_path}")
    return Phi

if __name__ == "__main__":
    run_simulation()