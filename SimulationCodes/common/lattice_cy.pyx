# lattice_cy.pyx
cimport numpy as np
import numpy as np
cdef double l_planck = 1.616229e-35
cdef double hbar_c = 1.973269804e-7
cdef double alpha = 1.0 / 137.035999084
cdef double beta = 2.19e-7
cdef double beta_su3 = 2.19e-7
cdef double kappa_vac = 2.61e-122
cdef double Lambda_eff = 1.0 / (2.0 * np.pi * l_planck)
cdef double lambda_5 = 0.13
cdef double lambda_56 = 0.055
cdef double lambda_ij = 0.055
cdef double v_5 = 246.0
cdef double mu_5 = -2 * lambda_5 * v_5**2
cdef double y_e = 2.14e-6
cdef double y_nu = 2.09e-10
cdef double zeta = 4.0e-4

def compute_action(np.ndarray[np.complex128_t, ndim=5] Phi, bint include_spin,
                   bint include_constraint, double a_t):
    cdef int Nx = Phi.shape[0], Ny = Phi.shape[1], Nz = Phi.shape[2], Nt = Phi.shape[3]
    cdef int N_fields = Phi.shape[4]
    cdef double kinetic = 0.0, V_planck = 0.0, V_ew = 0.0, V_qcd = 0.0
    cdef double V_yukawa = 0.0, V_baryon = 0.0, spin_su2 = 0.0, spin_su3 = 0.0
    cdef np.ndarray[np.double_t, ndim=4] Phi_norm = np.sqrt(np.sum(np.abs(Phi)**2, axis=-1).real)
    
    for i in range(4):
        kinetic += np.sum(np.abs(np.roll(Phi, -1, axis=i) - Phi)**2)
    kinetic *= (hbar_c/l_planck)**2
    V_planck = alpha * (hbar_c/l_planck)**2 * np.sum(Phi_norm / \
               (np.exp(Phi_norm * l_planck/hbar_c) - 1)) * a_t**3
    V_ew = mu_5 * np.sum(np.abs(Phi[...,4:6])**2) * a_t**3 + \
           lambda_5 * np.sum(np.abs(Phi[...,4:6])**4) * a_t**3 + \
           lambda_56 * np.sum(np.abs(Phi[...,4])**2 * np.abs(Phi[...,5])**2) * a_t**3
    V_qcd = lambda_ij * np.sum(np.abs(Phi[...,[2,3,6]])**2 * \
                              np.abs(Phi[...,[2,3,6]])**2) * a_t**3
    V_yukawa = y_e * np.sum((np.abs(Phi[...,0])**2 + np.abs(Phi[...,1])**2) * \
                            (np.sqrt(np.abs(Phi[...,4])**2) - v_5)) * a_t**3 + \
               y_nu * np.sum(np.abs(Phi[...,7])**2 * \
                            (np.sqrt(np.abs(Phi[...,4])**2) - v_5)) * a_t**3
    V_baryon = zeta * np.sum((Phi[...,2].conj() * Phi[...,3] - \
                              Phi[...,3].conj() * Phi[...,2] + \
                              Phi[...,6].conj() * Phi[...,3] - \
                              Phi[...,3].conj() * Phi[...,6]).imag) * \
                           np.sqrt(np.abs(Phi[...,4])**2) * a_t**3
    if include_spin:
        Phi_e = Phi[..., [0,1,7]]
        S_z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
        B_z = np.ones((Nx, Ny, Nz, Nt, 1)) * 1e-8
        spin_su2 = beta * np.sum(np.real(np.einsum('...i,ij,...j->...', \
                    np.conj(Phi_e), S_z, B_z * Phi_e)))
        Phi_q = Phi[..., [2,3,6]]
        Lambda_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
        B_3 = np.ones((Nx, Ny, Nz, Nt, 1)) * 1e-8
        spin_su3 = beta_su3 * np.sum(np.real(np.einsum('...i,ij,...j->...', \
                    np.conj(Phi_q), Lambda_3, B_3 * Phi_q)))
    if include_constraint:
        constraint = np.sum(np.abs(Phi[..., [2,3,4,5,6,7]])**2)
    return (kinetic - V_planck - V_ew - V_qcd - V_yukawa - V_baryon + \
            spin_su2 + spin_su3 + kappa_vac * (1e26**2 / (4 * l_planck**2))) * \
           (l_planck/hbar_c)**4

def compute_lorentz_ratio(np.ndarray[np.complex128_t, ndim=5] Phi):
    cdef double c = 2.99792458e8
    grad_mu = np.zeros_like(Phi)
    grad_i = np.zeros_like(Phi)
    for mu in range(4):
        grad_mu += (np.roll(Phi, -1, axis=mu) - Phi) / l_planck
    for i in range(3):
        grad_i += (np.roll(Phi, -1, axis=i) - Phi) / l_planck
    num = np.sum(np.abs(grad_mu)**2) / c**2
    denom = np.sum(np.abs(grad_i)**2)
    return num / denom if denom != 0 else 1.0