import numpy as np
from common_utils import metropolis_step

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
alpha = 1/137.035999084
g = 0.652
m_W = 80.379
Nx, Ny, Nz, Nt = 64, 64, 64, 256

Phi = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 8)) + \
      1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 8))

def compute_action_electroweak(Phi):
    kinetic = np.sum(np.abs(np.roll(Phi, -1, axis=(0,1,2,3)) - Phi)**2) / l_planck**2
    potential = g**2 * np.sum(np.abs(Phi[...,:4])**2) / (4 * l_planck**2)
    return (kinetic + potential) * l_planck**4

def run_electroweak_simulation():
    global Phi
    for step in range(5000):
        Phi, S = metropolis_step(Phi, compute_action_electroweak, 0.01)
        if step % 100 == 0:
            print(f"Step {step}, Action: {S:.2e}")
    W_mass = np.sqrt(g**2 * np.mean(np.abs(Phi[...,:4])**2)) * hbar_c / l_planck
    print(f"W boson mass: {W_mass:.3f} GeV, Error: {abs(W_mass - m_W)/m_W*100:.2f}%")
    np.save("SimulationCodes/paper2/data/Phi_electroweak.npy", Phi)
    np.save("SimulationCodes/paper2/data/W_mass.npy", np.array(W_mass))

if __name__ == "__main__":
    run_electroweak_simulation()