cimport numpy as np
import numpy as np
cdef double l_planck = 1.616229e-35
cdef double hbar_c = 1.973269804e-7
cdef double c = 2.99792458e8
cdef double alpha = 1/137.035999084
cdef double g = 0.652

def compute_action(
    double l_planck, double mu_sq, double lambda_, double epsilon, double alpha,
    double gamma, double beta, double kappa_vac, double xi,
    int Nx, int Ny, int Nz, int Nt, np.ndarray[np.complex128_t, ndim=5] Phi
):
    cdef double kinetic = 0.0, potential = 0.0
    cdef int d
    cdef np.ndarray[np.complex128_t, ndim=5] grad = np.zeros_like(Phi)
    for d in range(4):
        grad += np.abs(np.roll(Phi, -1, axis=d) - Phi)**2 / l_planck**2
    kinetic = np.sum(grad).real
    potential = g**2 * np.sum(np.abs(Phi[...,:4])**2) / (4 * l_planck**2)
    return (kinetic + potential) * l_planck**4