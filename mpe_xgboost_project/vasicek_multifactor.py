
import numpy as np

def simulate_multifactor_vasicek(r0_vec, a, b_vec, sigma_vec, T, dt, n_paths):
    n_factors = len(r0_vec)
    n_steps = int(T / dt)
    rates = np.zeros((n_paths, n_steps, n_factors))
    rates[:, 0, :] = r0_vec

    for t in range(1, n_steps):
        z = np.random.normal(size=(n_paths, n_factors))
        rates[:, t, :] = (
            rates[:, t-1, :]
            + a * (b_vec - rates[:, t-1, :]) * dt
            + sigma_vec * np.sqrt(dt) * z
        )
    return rates
