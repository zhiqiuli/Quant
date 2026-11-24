"""Demo: fit the two-factor Andersen vol curve to SYNTHETIC data and plot it.

Two made-up curves are generated, each from its own parameters plus noise:
  - synthetic ATM implied vols (high, steeply decaying), and
  - synthetic historical vols (lower, as implied vols usually sit above realised vols).
No market data is used.
Usage:  python Python/demo_synthetic_fit.py      (writes Documentation/demo_fit.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from andersen_model import AndersenParams, calibrate, sigma12

OUT = Path(__file__).resolve().parent.parent / "Documentation" / "demo_fit.png"

TRUE_ATM = AndersenParams(sig0=0.65, sig_inf=0.27, kappa=1.0, rho_inf=0.5)
TRUE_HIST = AndersenParams(sig0=0.40, sig_inf=0.10, kappa=1.6, rho_inf=0.4)


def synthetic_curve(true, T, rng, noise=0.006):
    return sigma12(T, true) + rng.normal(0, noise, T.size)


def main(seed=7):
    rng = np.random.default_rng(seed)
    T = np.linspace(0.05, 2.0, 24)                      # 24 monthly tenors, in years
    atm = synthetic_curve(TRUE_ATM, T, rng)
    hist = synthetic_curve(TRUE_HIST, T, rng)

    fit_atm, rmse_atm = calibrate(T, atm, rho_inf=TRUE_ATM.rho_inf)
    fit_hist, rmse_hist = calibrate(T, hist, rho_inf=TRUE_HIST.rho_inf)
    for name, true, fit, rmse in [("ATM ", TRUE_ATM, fit_atm, rmse_atm), ("hist", TRUE_HIST, fit_hist, rmse_hist)]:
        print(f"{name} true: sig0={true.sig0:.3f} sig_inf={true.sig_inf:.3f} kappa={true.kappa:.3f}")
        print(f"{name} fit : sig0={fit.sig0:.3f} sig_inf={fit.sig_inf:.3f} kappa={fit.kappa:.3f}  rmse={rmse:.4f}")

    grid = np.linspace(0, 2.0, 200)
    assert np.all(sigma12(grid, fit_atm) > sigma12(grid, fit_hist)), "implied should sit above historical"

    blue, orange = "#1f77b4", "#d95f02"
    fig, ax = plt.subplots(figsize=(8, 6))                 # 4:3 figure and plot area, same look as the full project
    ax.set_box_aspect(3 / 4)
    ax.plot(T, atm, "^", ms=5, color=blue, label="ATM vol")
    ax.plot(grid, sigma12(grid, fit_atm), "-", lw=1.6, color=blue, label="Fitted ATM")
    ax.plot(T, hist, "o", ms=4, color=orange, label="Hist vol")
    ax.plot(grid, sigma12(grid, fit_hist), "--", lw=1.6, color=orange, label="Fitted hist")
    ax.set_xlabel("contract (in years)")
    ax.set_ylabel("vol")
    ax.set_title("Two-factor Andersen curves fitted to synthetic data", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
