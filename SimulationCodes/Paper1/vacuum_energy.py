# vacuum_energy.py
import numpy as np
from scipy import integrate

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
kappa_vac = 2.61e-122
d_universe = 1e26

def integrand(k):
    S_ent = (d_universe)**2 / (4 * l_planck**2)
    Lambda_d = 1 / (l_planck * (1 + kappa_vac * S_ent))
    return (hbar_c * k / 2) * (1 / (np.exp(k / Lambda_d) - 1)) * \
           4 * np.pi * k**2 / (2 * np.pi)**3

rho_vac, err = integrate.quad(integrand, 0, 1 / (2 * np.pi * l_planck))
print(f"Vacuum energy: {rho_vac:.3e} GeV^4, Error: {err:.2e}")