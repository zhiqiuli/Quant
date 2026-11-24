# Two-factor Andersen model for commodity vol surfaces

A showcase of a project that fits the two-factor Andersen model (Andersen, 2010) to commodity
implied and historical vol curves for potential-future-exposure risk-factor modelling.

This folder contains a short overview and a few demos that run on **synthetic data**. The full
implementation (market-data pipeline, calibration for WTI, Brent and Henry Hub, seasonality) is available on request.

## Contents

| Path | What |
|---|---|
| `Documentation/two_factor_andersen_model_page1.pdf` | First page of the model notes: model definition and parameter meaning |
| `Documentation/henry_hub_simulation.png` | Henry Hub front-month Monte Carlo simulation (2.5% / 97.5% percentiles, median, initial curve) from the full implementation, 5000 paths, with seasonality |
| `Python/andersen_model.py` | Model vol curve and a simple least-squares fit |
| `Python/demo_synthetic_fit.py` | Fits synthetic ATM implied and historical vol curves (implied above historical) and writes `Documentation/demo_fit.png` |
| `Python/tests/test_andersen_model.py` | Offline tests of the curve limits and parameter recovery |

## Run

```bash
pip install -r requirements.txt
python Python/demo_synthetic_fit.py
python Python/tests/test_andersen_model.py
```

![Synthetic demo](Documentation/demo_fit.png)

Henry Hub front-month simulation (full implementation, 2026-10-30 calibration):

![Henry Hub simulation](Documentation/henry_hub_simulation.png)

Reference: Andersen, L. (2010). Markov models for commodity futures: theory and practice.
Quantitative Finance, 10(8), 831-854.
