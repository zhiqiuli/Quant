"""Demo: fit the two-factor Andersen vol curve to SYNTHETIC data and plot it.

No market data is used: the "observed" vols are generated from made-up parameters plus noise.
Usage:  python Python/demo_synthetic_fit.py      (writes Documentation/demo_fit.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from andersen_model import AndersenParams, calibrate, sigma12

OUT = Path(__file__).resolve().parent.parent / "Documentation" / "demo_fit.png"


def main(seed=7):
    rng = np.random.default_rng(seed)
    true = AndersenParams(sig0=0.60, sig_inf=0.25, kappa=1.2, rho_inf=0.5)
    T = np.linspace(0.05, 2.0, 24)                      # 24 monthly tenors, in years
    obs = sigma12(T, true) + rng.normal(0, 0.006, T.size)

    fit, rmse = calibrate(T, obs, rho_inf=true.rho_inf)
    print(f"true : sig0={true.sig0:.3f} sig_inf={true.sig_inf:.3f} kappa={true.kappa:.3f}")
    print(f"fit  : sig0={fit.sig0:.3f} sig_inf={fit.sig_inf:.3f} kappa={fit.kappa:.3f}  rmse={rmse:.4f}")

    grid = np.linspace(0, 2.0, 200)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(T, obs, "^", ms=5, color="tab:blue", label="synthetic vols")
    ax.plot(grid, sigma12(grid, fit), "-", color="tab:blue", label="fitted")
    ax.set_xlabel("contract (in years)")
    ax.set_ylabel("vol")
    ax.set_title("Two-factor Andersen curve fitted to synthetic data", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
