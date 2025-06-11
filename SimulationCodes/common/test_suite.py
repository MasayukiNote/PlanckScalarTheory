# test_suite.py
import pytest
import numpy as np
from Paper1 import vacuum_energy, lamb_shift, black_body_simulation, sg_effect, g2_simulation, ab_effect
import lattice_cy

def test_vacuum_energy():
    l_planck = 1.616229e-35
    hbar_c = 1.973269804e-7
    kappa_vac = 2.61e-122
    d_universe = 1e26
    S_ent = (d_universe)**2 / (4 * l_planck**2)
    Lambda_d = 1 / (l_planck * (1 + kappa_vac * S_ent))
    rho_vac, _ = vacuum_energy.integrate.quad(vacuum_energy.integrand, 0, 1 / (2 * np.pi * l_planck))
    assert abs(rho_vac - 1.080e-47) / 1.080e-47 < 0.005, "Vacuum energy deviates >0.5%"

def test_lamb_shift():
    freq = lamb_shift.compute()  # Assuming lamb_shift.py has a compute() function
    assert abs(freq - 1058.8) / 1058.8 < 0.0007, "Lamb shift deviates >0.07%"

def test_black_body():
    T = 6000
    u_simulated = black_body_simulation.compute_u(T)  # Assuming compute_u() function
    u_theoretical, _ = black_body_simulation.integrate.quad(lambda nu: black_body_simulation.u_nu(nu, T), 0, 1e20)
    assert abs(u_simulated - u_theoretical) / u_theoretical < 0.001, "Black-body deviates >0.1%"

def test_sg_effect():
    theta = sg_effect.compute_sg_deflection(sg_effect.Phi_e)
    assert abs(theta - 0.01) / 0.01 < 0.05, "SG deflection deviates >5%"

def test_g2():
    Delta_a = g2_simulation.compute_g2()  # Assuming compute_g2() function
    assert abs(Delta_a - 0.001159652) / 0.001159652 < 0.00048, "Muon g-2 deviates >0.048%"

def test_ab_effect():
    Delta_theta = ab_effect.compute_ab_phase(4.135e-15)
    assert abs(Delta_theta - 6.283) / 6.283 < 1e-3, "AB phase deviates >0.1%"

def test_lattice_action():
    Nx, Ny, Nz, Nt, N_fields = 4, 4, 4, 4, 8
    Phi = np.random.normal(0, 1.973269804e-7/1.616229e-35, (Nx, Ny, Nz, Nt, N_fields)) + \
          1j * np.random.normal(0, 1.973269804e-7/1.616229e-35, (Nx, Ny, Nz, Nt, N_fields))
    action = lattice_cy.compute_action(Phi, False, False, 1.0)
    assert action > 0, "Action negative"