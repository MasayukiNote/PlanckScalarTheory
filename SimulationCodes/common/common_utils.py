import numpy as np

def metropolis_step_common(phi, action_func, step_size=0.01, l_planck=1.616229e-35):
    phi_new = phi.copy()
    dims = phi.shape
    idx = tuple(np.random.randint(0, dim) for dim in dims[:-1]) + (np.random.randint(0, dims[-1]),)
    phi_new[idx] += np.random.normal(0, step_size * l_planck) + \
                    1j * np.random.normal(0, step_size * l_planck)
    S_old = action_func(phi)
    S_new = action_func(phi_new)
    if S_new <= S_old or np.random.random() < np.exp(-(S_new - S_old)):
        return phi_new, S_new
    return phi, S_old