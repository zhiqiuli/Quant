
import numpy as np

def discount_factors_from_forwards(forwards, dt=0.25):
    dfs = []
    acc = 1.0
    for f in forwards:
        acc *= np.exp(-f * dt)
        dfs.append(acc)
    return np.array(dfs)

def compute_par_swap_rate(discount_forwards, maturity, freq=4):
    dt = 1 / freq
    n_periods = int(maturity * freq)
    dfs = discount_factors_from_forwards(discount_forwards[:n_periods], dt)
    numerator = 1 - dfs[-1]
    denominator = np.sum(dfs * dt)
    return numerator / denominator

def compute_dv01(discount_forwards, maturity, freq=4):
    dt = 1 / freq
    n_periods = int(maturity * freq)
    dfs = discount_factors_from_forwards(discount_forwards[:n_periods], dt)
    annuity = np.sum(dfs * dt)
    return annuity * 1e-4
