cimport numpy as np
import numpy as np
cdef double l_planck = 1.616229e-35
cdef double hbar_c = 1.973269804e-7
cdef double c = 2.99792458e8
cdef double alpha = 1/137.035999084

cdef np.ndarray[np.complex128_t, ndim=2] S_z = np.array(
    [[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128) * hbar_c / (2 * np.pi)
cdef np.ndarray[np.complex128_t, ndim=2] S_x = np.array(
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128) * hbar_c / (2 * np.sqrt(2) * np.pi)
cdef np.ndarray[np.complex128_t, ndim=2] S_y = np.array(
    [[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128) * hbar_c / (2 * np.sqrt(2) * np.pi)

def compute_action(
    double l_planck, double mu_sq, double lambda_, double epsilon, double alpha,
    double gamma, double beta, double kappa_vac, double xi,
    int Nx, int Ny, int Nz, int Nt, np.ndarray[np.complex128_t, ndim=5] Phi,
    str paper_type
):
    cdef double kinetic = 0.0, potential = 0.0, spin = 0.0
    cdef int i, j, k, t, d
    cdef np.ndarray[np.complex128_t, ndim=5] grad = np.zeros_like(Phi)

    # Kinetic term
    for d in range(4):
        grad += np.abs(np.roll(Phi, -1, axis=d) - Phi)**2 / l_planck**2
    kinetic = np.sum(grad).real

    if paper_type in ["paper1", "paper3"]:
        # Potential term
        Phi_norm = np.sqrt(np.sum(np.abs(Phi[..., :3])**2, axis=-1))
        exponent = Phi_norm * l_planck
        potential = (alpha / l_planck**2) * np.sum(Phi_norm / (np.exp(exponent) - 1))
        # Spin term
        Phi_gamma = Phi[..., :3]
        spin_term = np.einsum('...i,ij,...j->...', np.conj(Phi_gamma), S_z, Phi_gamma).real
        spin = beta * np.sum(spin_term)
        # Total action
        return (kinetic - mu_sq * np.sum(np.abs(Phi)**2) + lambda_ * np.sum(np.abs(Phi)**4) +
                epsilon * np.sum(np.abs(Phi)**2 / (np.exp(Phi_norm) - 1)**gamma) +
                potential + spin - kappa_vac * np.sum(np.abs(Phi)**2) - xi * np.sum(np.abs(Phi)**4)) * l_planck**4
    elif paper_type == "paper2":
        # Electroweak potential
        potential = (0.652**2 * np.sum(np.abs(Phi[...,:4])**2)) / (4 * l_planck**2)
        return (kinetic + potential) * l_planck**4
    else:
        raise ValueError(f"Unknown paper_type: {paper_type}")