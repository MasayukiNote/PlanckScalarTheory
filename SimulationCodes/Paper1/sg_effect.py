# sg_effect.py
import numpy as np

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
beta = 2.19e-7
Nx, Ny, Nz, Nt = 48, 48, 48, 96

Phi_e = np.random.normal(0, hbar_c/l_planck, (Nx, Ny, Nz, Nt, 3)) + \
        1j * np.random.normal(0, hbar_c/l_planck, (Nx, Ny, Nz, Nt, 3))

def compute_sg_deflection(Phi):
    S_z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    B_z_grad = 1.0  # T/m
    S_z_exp = np.mean(np.einsum('...i,ij,...j->...', np.conj(Phi), S_z, Phi)).real
    F_z = beta * S_z_exp * B_z_grad
    p = 0.51099895000e-3 * 2.99792458e8 / hbar_c
    l = 1e-3
    theta = F_z * l / (p * 2.99792458e8)
    return theta

theta = compute_sg_deflection(Phi_e)
print(f"SG deflection angle: {theta:.3e} rad")