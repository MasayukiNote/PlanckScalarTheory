# lamb_shift.py
import numpy as np
from scipy import integrate

alpha = 1 / 137.035999084
l_planck = 1.616229e-35
m_e = 0.51099895000e-3
Lambda_eff = 1 / (2 * np.pi * l_planck)

def integrand(k, x):
    return (2 * m_e / ((k**2 - x * m_e**2)**2)) * (alpha / (4 * np.pi)) * \
           (1 / (np.exp(k * l_planck) - 1))

eta_spin, _ = integrate.quad(lambda x: integrate.quad(integrand, 0, Lambda_eff,
                                                     args=(x,))[0], 0, 1)
eta_vac = (8 / 8) * 1.14
eta = eta_spin * eta_vac
delta_E = eta * (alpha**5 * m_e * 2.99792458e8**2 / 6) * np.log(1 / alpha)
freq = delta_E / (6.582119569e-22) / 1e6
print(f"Lamb shift: {freq:.1f} MHz")