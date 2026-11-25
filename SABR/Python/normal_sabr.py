"""Normal SABR (beta = 0): Hagan normal implied vol, ATM-constrained calibration, Bachelier receiver.

These are the formulas on page 1 of Documentation/receiver_swaption_normal_sabr_page1.pdf.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm


def hagan_normal_vol(F, K, T, alpha, rho, nu):
    """Hagan normal implied vol for beta = 0."""
    F, K = np.asarray(F, float), np.asarray(K, float)
    z = nu * (F - K) / alpha
    x = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    # z/x -> 1 as z -> 0
    ratio = np.where(np.abs(z) < 1e-8, 1.0, z / np.where(np.abs(z) < 1e-8, 1.0, x))
    return alpha * ratio * (1 + (2 - 3 * rho**2) / 24 * nu**2 * T)


def alpha_from_atm(sigma_atm, T, rho, nu):
    """Closed-form alpha0 that matches the ATM normal vol exactly."""
    return sigma_atm / (1 + (2 - 3 * rho**2) / 24 * nu**2 * T)


def calibrate(F, T, sigma_atm, strikes, vols, x0=(-0.2, 0.4)):
    """Fit (rho, nu) to off-ATM quotes; alpha0 is re-solved to hit ATM at each trial."""
    strikes, vols = np.asarray(strikes, float), np.asarray(vols, float)

    def resid(p):
        rho, nu = p
        a = alpha_from_atm(sigma_atm, T, rho, nu)
        # Residuals in bp so the optimizer's tolerances are not hit early on decimal-vol scale.
        return (hagan_normal_vol(F, strikes, T, a, rho, nu) - vols) / 1e-4

    sol = least_squares(resid, x0, bounds=([-0.999, 1e-6], [0.999, 5.0]), xtol=1e-14, ftol=1e-14)
    rho, nu = sol.x
    return rho, nu, alpha_from_atm(sigma_atm, T, rho, nu), sol


def bachelier_receiver(F, K, T, sigma, N, A):
    """Physically settled receiver swaption (put on the forward swap rate) under Bachelier."""
    d = (K - F) / (sigma * np.sqrt(T))
    return N * A * ((K - F) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d))
