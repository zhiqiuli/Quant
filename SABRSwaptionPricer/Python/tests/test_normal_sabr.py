"""Offline checks of the normal SABR vol, the calibration and the Bachelier receiver."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normal_sabr import alpha_from_atm, bachelier_receiver, calibrate, hagan_normal_vol

BP = 1e-4


class NormalSabrTest(unittest.TestCase):
    F, T, RHO, NU, ATM = 0.04, 364 / 365, -0.35, 0.50, 60 * BP

    def alpha(self):
        return alpha_from_atm(self.ATM, self.T, self.RHO, self.NU)

    def test_atm_matches_quote(self):
        vol = float(hagan_normal_vol(self.F, self.F, self.T, self.alpha(), self.RHO, self.NU))
        self.assertAlmostEqual(vol, self.ATM, places=14)

    def test_smile_continuous_at_atm(self):
        # Strikes just inside and just outside the z -> 0 cutoff use different code branches.
        a = self.alpha()
        inside, outside = (float(hagan_normal_vol(self.F, self.F + h, self.T, a, self.RHO, self.NU))
                           for h in (1e-10, 2e-10))
        self.assertLess(abs(outside - inside), 1e-10)

    def test_negative_rho_gives_receiver_skew(self):
        a = self.alpha()
        low, high = hagan_normal_vol(self.F, np.array([self.F - 50 * BP, self.F + 50 * BP]),
                                     self.T, a, self.RHO, self.NU)
        self.assertGreater(low, self.ATM)
        self.assertLess(high, self.ATM)

    def test_calibration_recovers_parameters(self):
        strikes = np.array([self.F - 50 * BP, self.F + 50 * BP])
        quotes = hagan_normal_vol(self.F, strikes, self.T, self.alpha(), self.RHO, self.NU)
        for x0 in [(-0.2, 0.4), (0.0, 0.3), (-0.5, 0.8)]:
            rho, nu, _, _ = calibrate(self.F, self.T, self.ATM, strikes, quotes, x0=x0)
            self.assertAlmostEqual(rho, self.RHO, places=6)
            self.assertAlmostEqual(nu, self.NU, places=6)

    def test_bachelier_limits(self):
        N, A, K = 1e8, 4.4, 0.0375
        # Zero vol: intrinsic value. At the money: N A sigma sqrt(T / 2 pi).
        self.assertAlmostEqual(bachelier_receiver(0.03, K, 1.0, 1e-12, N, A), N * A * 0.0075, places=2)
        self.assertAlmostEqual(bachelier_receiver(0.04, 0.04, 1.0, self.ATM, N, A),
                               N * A * self.ATM / np.sqrt(2 * np.pi), places=6)


if __name__ == "__main__":
    unittest.main()
