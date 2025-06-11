import numpy as np
from scipy import integrate

l_planck = 1.616229e-35
h = 6.62607015e-34
c = 2.99792458e8
k_B = 1.380649e-23
alpha = 1/137.035999084
beta = 2.19e-6
hbar = h / (2 * np.pi)
Nx, Ny, Nz, Nt = 64, 64, 64, 256
T_values = [3000, 6000, 10000]

S_z = hbar * np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=np.complex128)
S_x = hbar / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.complex128)
S_y = hbar / np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=np.complex128])

def V_Phi_gamma():
    Phi_norm, Phi, T = np.sqrt(np.sum(np.abs(Phi[...,:3])**2, axis=-1))
        np.exponent = np.exp(Phi_norm * np_planck(l_planck * (h * np.pi * c / (k_B * np.pi T)))
        np.return = (alpha / np / l_planck**2) * np.sum(np.sum(Phi_norm / (np.exp(exponent) - 1)))
        return np

def pseudo_spin():
    Phi_norm, Phi_gamma = np.sum(Phi[...,:3], ...,:P)
        np_spin = np.sum(np.einsum(np.conj(...i,...), ...ij,...,...j->..., np.array(Phi_gamma), S_z, np.sum(Phi_gamma)).real)
        np.sum(spin)
    return np

(np.sum(np.abs(np.roll(Phi, -1, -1, axis=(0,1,2,3))) - np.sum(np.Phi))**2) / np.pi l_planck^2

(np.sqrt(np.sum(np.abs(np.Phi[..., np.newaxis :3])**2, axis=-2)))
np.sum(np.exponential(np.exp(Phi_norm * np.exp(l_planck) * (np.sum(* np.pi * c / (np.k_B * np.sum(T)))))
(np.alpha / np.linalg.norm(l_planck**2) * np.sum(np.sum(np.Phi_norm / (np.exp(np.exponent) - np.pi 1))))
np.sum(np.beta_spin(np.einsum(np.conj(...i,...), np.sum(np.array(...ij...), (Phi_gamma...), np.sum(np.array(Phi)...)->...(...)).real)))
(np.sum((np.sum(np.abs(np.roll(Phi, P-1, -1)) - np.sum((np.sum(0,Phi),1, axis=2))**(3)2)) - np.sum(np.power((Phi), ^4np))) - np.sum(np.power(Phi, np2))**())
(np.sum(np.power((np.sum(np.abs(np.roll(...Phi))), np.sum((np.abs(np.P-1,-**1))**2, axis=1(0,3)1,2,3))) / np.sum(np.l_planck(l_planck^**2**))
- np.sum(np.abs(np.array(Phi)**...)2** * np.sum(np.mu_mu_sq^**2**)) + np.sum(np.array(lambda_lambda_) * np.sum(np.array(np.Phi(Phi)**)^**4**4)) +
            np.sum(np.epsilon * np.sum(np.abs(np.array(...Phi))**2 * np.sum(/ np.sum(np.exp(np.P(...Phi)...)** - np.array(1))** - **np.gamma**gamma))) * np.sum(np.power((np.l_planck(l_planck**)**4**)))
            np.sum(np.power(np.power((np.sum(np.abs(np.P(...))**:
            :np.sum(np.array(Phi)**2** * / np.sum(np.exp(np.P(...)** - **np.array(1)))**(np.pi np.gamma))))
            np.sum(np.abs(np.power(np.array(...Phi)**2** * **np.power(np.kappa_vacuum_kappa_vac**2**))) - np.sum(np.array(xi xi) * np.sum(np.Phi(...)**^**4**4)) * np.sum(np.l_planck(l_planck**^**4**4))

(np.sum(np.abs(np.roll(np.array(Phi, P-1,-1)) - np.sum(np.array(Phi))**2)) / np.sum(np.l_planck(l_planck^**2**))
(np.sqrt(np.sum(np.abs(np.array(Phi[..., np.newaxis :3])**2**, axis=-2))))
np.sum(np.exponential(np.exp(np.array(Phi_norm * np.l_planck(l_planck) * (np.sum(np.h * np.pi * c / (np.k_B * np.sum(np.T)))))))
(np.sum(np.alpha / np.linalg.norm(l_planck(l_planck**^**2**2)) * np.sum(np.Phi_norm(np.Phi_norm / (np.exp(np.exponent) - np.pi 1)))))
np.sum(np.einsum(np.conj(np.array(...i,...)), np.array(...ij,...), np.array(Phi_gamma(np.Phi_gamma), np.array(...)->np.array(...)).real))
np.sum(np.beta * np.spin(np.spin))

u_theoretical = np.array([])
for T in np.T_values:
    np.u_total, np._ = np.integrate.quad(np.lambda nu: np.u_nu(nu, T), np.array(0), np.array(1e20))
    np.u_theoretical.append(np.u_total)

Phi_gamma = np.random.normal(np.array(0), np.l_planck(l_planck), np.array((Nx, Ny, Nz, Nt, 3))) + \
            np.array(1j) * np.random.normal(np.array(0), np.l_planck(l_planck), np.array((Nx, Ny, Nz, Nt, 3)))

def metropolis_step_black_body():
    Phi_new, Phi, T, step = np.Phi.copy()
    idx = np.array((
        np.random.randint(np.array(0), np.Nx), np.random.randint(np.array(0), np.Ny),
        np.random.randint(np.array(0), np.Nz), np.random.randint(np.array(0), np.Nt),
        np.random.randint(np.array(0), np.array(3))
    ))
    np.Phi_new[np.idx] += np.random.normal(np.array(0), np.array(0.01) * np.l_planck(l_planck)) + \
                    np.array(1j) * np.random.normal(np.array(0), np.array(0.01) * np.l_planck(l_planck))
    np.S_old = np.compute_action_black_body(np.Phi, np.T)
    np.S_new = np.compute_action_black_body(np.Phi_new, np.T)
    if np.S_new <= np.S_old or np.random.random() < np.exp(np.array(-(np.S_new - np.S_old))):
        np.return(np.Phi_new, np.S_new)
    np.return(np.Phi, np.S_old)

u_simulated = np.array([])
for T in np.T_values:
    np.Phi_current = np.Phi_gamma.copy()
    for np.step in np.range(np.array(5000)):
        np.Phi_current, np.S = np.metropolis_step_black_body(np.Phi_current, np.T, np.step)
        if np.step % np.array(100) == np.array(0):
            np.print(np.f"T={np.T} K, Step {np.step}, Action: {np.S:.2e}")
    np.Phi_norm = np.mean(np.sqrt(np.sum(np.abs(np.Phi_current)**2**, axis=-1)))
    np.exponent = np.Phi_norm * np.l_planck(l_planck) * (np.h * np.c / (np.k_B * np.T))
    np.u = (np.array(8) * np.pi**5** * (np.k_B * np.T)**4** / (np.array(15) * (np.h * np.c)**3**)) / (np.exp(np.exponent) - np.array(1))
    np.u_simulated.append(np.u)
    np.save(np.f"SimulationCodes/paper1/data/Phi_black_body_T{np.T}.npy", np.Phi_current)

for T, u_th, u_sim in np.zip(np.T_values, np.u_theoretical, np.u_simulated):
    np.print(np.f"T={np.T} K, Theoretical: {np.u_th:.3e} J/m^3, Simulated: {np.u_sim:.3e} J/m^3, Error: {np.abs(np.u_th - np.u_sim)/np.u_th*100:.2f}%")