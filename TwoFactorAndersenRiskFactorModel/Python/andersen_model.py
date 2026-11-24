"""Two-factor Andersen model: parametric vol curve (showcase version).

With a(T)=d(T)=0 the model vol curve is

    h1 = -sig_inf + rho_inf * sig0,   h_inf = sig_inf,   h2 = sig0 * sqrt(1 - rho_inf^2)
    sig12^2(T) = (h1^2 + h2^2) e^{-2 kappa T} + 2 h1 h_inf e^{-kappa T} + h_inf^2

so sig12(0) = sig0 (short-tenor vol) and sig12(T -> inf) = sig_inf (long-tenor vol);
kappa is the speed of the decay between the two.

Reference: Andersen, L. (2010), Markov models for commodity futures: theory and practice,
Quantitative Finance 10(8), 831-854.

This is a simplified demo: a generic least-squares fit with a naive starting point.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class AndersenParams:
    sig0: float
    sig_inf: float
    kappa: float
    rho_inf: float

    @property
    def h1(self):
        return -self.sig_inf + self.rho_inf * self.sig0

    @property
    def h2(self):
        return self.sig0 * np.sqrt(1.0 - self.rho_inf**2)

    @property
    def h_inf(self):
        return self.sig_inf


def sigma12(T, p: AndersenParams):
    """Model vol curve sigma_12(T)."""
    e = np.exp(-p.kappa * np.asarray(T, dtype=float))
    var = (p.h1**2 + p.h2**2) * e**2 + 2 * p.h1 * p.h_inf * e + p.h_inf**2
    return np.sqrt(np.maximum(var, 0.0))


def calibrate(T, vols, rho_inf=0.5):
    """Least-squares fit of (sig0, sig_inf, kappa) to a vol curve, with rho_inf held fixed.

    Returns (AndersenParams, rmse).
    """
    T, vols = np.asarray(T, dtype=float), np.asarray(vols, dtype=float)
    order = np.argsort(T)
    T, vols = T[order], vols[order]

    def resid(x):
        return sigma12(T, AndersenParams(x[0], x[1], x[2], rho_inf)) - vols

    x0 = [vols[0], vols[-1], 1.0]  # naive start: first vol, last vol, unit decay speed
    sol = least_squares(resid, x0, bounds=([1e-4, 1e-4, 1e-4], [5.0, 5.0, 50.0]))
    p = AndersenParams(*sol.x, rho_inf)
    return p, float(np.sqrt(np.mean(resid(sol.x) ** 2)))
