"""Offline checks of the vol-curve formula and the calibration (synthetic data only)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from andersen_model import AndersenParams, calibrate, sigma12


class AndersenModelTest(unittest.TestCase):
    P = AndersenParams(sig0=0.6, sig_inf=0.25, kappa=1.2, rho_inf=0.5)

    def test_limits(self):
        self.assertAlmostEqual(float(sigma12(0.0, self.P)), 0.6, places=12)      # short end = sig0
        self.assertAlmostEqual(float(sigma12(50.0, self.P)), 0.25, places=9)     # long end = sig_inf

    def test_recovers_parameters(self):
        T = np.linspace(0.05, 2.0, 24)
        fit, rmse = calibrate(T, sigma12(T, self.P), rho_inf=self.P.rho_inf)
        self.assertLess(rmse, 1e-6)
        for got, want in [(fit.sig0, 0.6), (fit.sig_inf, 0.25), (fit.kappa, 1.2)]:
            self.assertAlmostEqual(got, want, places=4)


if __name__ == "__main__":
    unittest.main()
