# ab_effect.py
import numpy as np

l_planck = 1.616229e-35
hbar_c = 1.973269804e-7
e = 1.602176634e-19
hbar = 6.62607015e-34 / (2 * np.pi)

def compute_ab_phase(Phi_B):
    Delta_theta = (e / hbar) * Phi_B
    return Delta_theta

Phi_B = 4.135e-15  # Weber
Delta_theta = compute_ab_phase(Phi_B)
print(f"AB phase shift: {Delta_theta:.3f}")