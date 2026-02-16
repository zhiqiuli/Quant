
import numpy as np
import pandas as pd
from vasicek_multifactor import simulate_multifactor_vasicek
from swap_analytics import compute_par_swap_rate, compute_dv01

TENORS = ["0d3m","3m6m","6m1y","1y2y","2y5y","5y10y","10y20y","20y30y"]

def swap_mtm_normalized(fwd_rates, fixed_rate, maturity):
    swap_rate = np.mean(fwd_rates, axis=1)
    mtm = (swap_rate - fixed_rate) * maturity
    return mtm

def compute_mpe_multifactor(fwd_init, fixed_rate, maturity):
    dt = 0.25
    n_paths = 3000
    a = 0.1
    b_vec = fwd_init
    sigma_vec = np.linspace(0.01, 0.02, len(fwd_init))

    rates = simulate_multifactor_vasicek(
        r0_vec=fwd_init,
        a=a,
        b_vec=b_vec,
        sigma_vec=sigma_vec,
        T=maturity,
        dt=dt,
        n_paths=n_paths
    )

    exposures = []
    for t in range(rates.shape[1]):
        mtm = swap_mtm_normalized(rates[:, t, :], fixed_rate, maturity)
        exposure = np.maximum(mtm, 0)
        exposures.append(np.percentile(exposure, 95))

    return np.max(exposures)

def generate_dataset(n_samples=800):
    data = []
    for _ in range(n_samples):
        maturity = np.random.uniform(5, 10)
        fixed_rate = np.random.uniform(0.00, 0.08)

        base_curve = np.linspace(0.03, 0.05, len(TENORS))
        noise = np.random.uniform(-0.005, 0.005, len(TENORS))
        fwd_init = base_curve + noise

        base_discount_curve = np.linspace(0.03, 0.05, 40)
        noise_discount = np.random.uniform(-0.005, 0.005, 40)
        discount_curve = base_discount_curve + noise_discount

        par_rate = compute_par_swap_rate(discount_curve, maturity)
        dv01 = compute_dv01(discount_curve, maturity)

        par_minus_fixed = par_rate - fixed_rate

        mpe = compute_mpe_multifactor(fwd_init, fixed_rate, maturity)

        row = list(fwd_init) + [
            maturity,
            par_minus_fixed,
            dv01,
            mpe
        ]

        data.append(row)

    columns = (
        TENORS +
        ["maturity",
         "par_minus_fixed",
         "dv01",
         "mpe_normalized"]
    )

    return pd.DataFrame(data, columns=columns)
