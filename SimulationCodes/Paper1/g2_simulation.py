# g2_simulation.py
import numpy as np
from scipy import integrate

alpha = 1 / 137.035999084
l_planck = 1.616229e-35
m_mu = 0.1056583755
Lambda_eff = 1 / (2 * np.pi * l_planck)

def integrand(k, x):
    return (2 * m_mu / ((k**2 - x * m_mu**2)**2)) * (alpha / (4 * np.pi)) * \
           (1 / (np.exp(k * l_planck) - 1))

eta_spin, _ = integrate.quad(lambda x: integrate.quad(integrand, 0, Lambda_eff,
                                                     args=(x,))[0], 0, 1)
eta_vac = (8 / 8) * 1.14
Delta_a = (alpha / (2 * np.pi)) * eta_spin * eta_vac
print(f"Muon g-2: {Delta_a:.9f}")