# Receiver swaption pricing with normal SABR

A showcase of a project that prices European, physically settled USD receiver swaptions with the
normal SABR model (β = 0). It uses the Hagan normal implied vol and the Bachelier formula.

This folder contains the first page of the note and the core formulas, with offline tests. The full
note (1Y × 5Y receiver worked example off a SOFR OIS curve, DV01 / gamma / vega, and a 2020 Covid
stress scenario on the curve and the vol surface) is available on request.

## Contents

| Path | What |
|---|---|
| `Documentation/receiver_swaption_normal_sabr_page1.pdf` | First page of the note: model dynamics, Hagan normal vol, Bachelier receiver, and parameter roles |
| `Python/normal_sabr.py` | Hagan normal vol, ATM-constrained (ρ, ν) calibration, Bachelier receiver |
| `Python/tests/test_normal_sabr.py` | Offline tests of the ATM fit, smile continuity and skew, parameter recovery, and Bachelier limits |

## Run

```bash
pip install -r requirements.txt
python Python/tests/test_normal_sabr.py
```

Reference: Hagan, P., Kumar, D., Lesniewski, A., Woodward, D. (2002). Managing smile risk.
Wilmott Magazine, September, 84-108.
