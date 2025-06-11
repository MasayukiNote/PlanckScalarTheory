import numpy as np

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
alpha = 1/137.035999084
Nx, Ny, Nz, Nt = 64, 64, 64, 256

Phi = np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6)) + \
      1j * np.random.normal(0, l_planck, (Nx, Ny, Nz, Nt, 6))

def compute_lamb_shift(Phi):
    Phi_norm = np.sqrt(np.sum(np.abs(Phi[...,:3])**2, axis=-1))
    delta_E = (alpha**5 / (4 * np.pi**2)) * (hbar_c / l_planck) * np.mean(Phi_norm)
    return delta_E * 2.418e14  # Convert to MHz

lamb_shift = compute_lamb_shift(Phi)
print(f"Lamb shift: {lamb_shift:.1f} MHz, Error: {abs(lamb_shift - 1058.8)/1058.8*100:.2f}%")
np.save("SimulationCodes/paper1/data/lamb_shift.npy", np.array(lamb_shift))