import numpy as np


def price_relative_perf_fx_trigger_mc(
    notional: float = 12_886_000,
    premium: float = 249_988.40,

    # Initial levels
    spx_0: float = 7117.5,
    sx5e_0: float = 5889.0,
    eurusd_0: float = 1.1572,

    # Forward levels to option expiry
    spx_fwd: float = 7198.25,
    sx5e_fwd: float = 5878.0,
    eurusd_fwd: float = 1.176,

    # Implied vols
    spx_vol: float = 0.176,
    sx5e_vol: float = 0.180,
    eurusd_vol: float = 0.063,

    # Correlations
    corr_spx_sx5e: float = 0.399,
    corr_spx_eurusd: float = 0.094,
    corr_sx5e_eurusd: float = 0.127,

    # Time
    T_option: float = 148 / 365,
    T_discount: float = 152 / 365,

    # Discount rate
    sofr_rate: float = 0.0368,

    # MC settings
    n_paths: int = 1_000_000,
    seed: int = 42,
):
    """
    Price payoff:

        Payoff = Notional
                 * 1{EURUSD_T < 0.99 * EURUSD_0}
                 * max(SPX_T / SPX_0 - SX5E_T / SX5E_0, 0)

    Each risk factor follows correlated lognormal terminal dynamics.
    Forward levels are used so that E[S_T] = F.
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Correlation matrix
    # -----------------------------
    corr = np.array([
        [1.0,            corr_spx_sx5e,   corr_spx_eurusd],
        [corr_spx_sx5e,  1.0,             corr_sx5e_eurusd],
        [corr_spx_eurusd, corr_sx5e_eurusd, 1.0]
    ])

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals <= 0):
        raise ValueError(f"Correlation matrix is not positive definite. Eigenvalues: {eigvals}")

    chol = np.linalg.cholesky(corr)

    # -----------------------------
    # 2. Simulate correlated normals
    # -----------------------------
    z_uncorr = rng.standard_normal(size=(n_paths, 3))
    z_corr = z_uncorr @ chol.T

    z_spx = z_corr[:, 0]
    z_sx5e = z_corr[:, 1]
    z_eurusd = z_corr[:, 2]

    # -----------------------------
    # 3. Simulate terminal levels
    # -----------------------------
    sqrt_T = np.sqrt(T_option)

    spx_T = spx_fwd * np.exp(
        -0.5 * spx_vol ** 2 * T_option
        + spx_vol * sqrt_T * z_spx
    )

    # Quanto-adjusted SX5E forward
    sx5e_quanto_fwd = sx5e_fwd * np.exp(
        -corr_sx5e_eurusd * sx5e_vol * eurusd_vol * T_option
    )

    sx5e_T = sx5e_quanto_fwd * np.exp(
        -0.5 * sx5e_vol ** 2 * T_option
        + sx5e_vol * sqrt_T * z_sx5e
    )

    eurusd_T = eurusd_fwd * np.exp(
        -0.5 * eurusd_vol ** 2 * T_option
        + eurusd_vol * sqrt_T * z_eurusd
    )

    # -----------------------------
    # 4. Payoff calculation
    # -----------------------------
    spx_perf = spx_T / spx_0
    sx5e_perf = sx5e_T / sx5e_0

    eurusd_trigger = 0.99 * eurusd_0

    fx_trigger_hit = eurusd_T < eurusd_trigger

    payoff = notional * fx_trigger_hit * np.maximum(spx_perf - sx5e_perf, 0.0)

    # -----------------------------
    # 5. Discounting
    # -----------------------------
    discount_factor = np.exp(-sofr_rate * T_discount)

    pv_paths = discount_factor * payoff

    price = np.mean(pv_paths)
    std_error = np.std(pv_paths, ddof=1) / np.sqrt(n_paths)

    # Useful diagnostics
    undiscounted_expected_payoff = np.mean(payoff)
    trigger_probability = np.mean(fx_trigger_hit)
    positive_equity_payoff_probability = np.mean(spx_perf > sx5e_perf)
    positive_total_payoff_probability = np.mean(payoff > 0)

    premium_pct = premium / notional
    price_pct = price / notional

    results = {
        "price": price,
        "price_pct_notional": price_pct,
        "std_error": std_error,
        "std_error_pct_notional": std_error / notional,
        "undiscounted_expected_payoff": undiscounted_expected_payoff,
        "discount_factor": discount_factor,
        "trigger_probability": trigger_probability,
        "positive_equity_payoff_probability": positive_equity_payoff_probability,
        "positive_total_payoff_probability": positive_total_payoff_probability,
        "dealer_premium": premium,
        "dealer_premium_pct_notional": premium_pct,
        "model_minus_premium": price - premium,
        "model_minus_premium_pct_notional": price_pct - premium_pct,
        "correlation_eigenvalues": eigvals,
    }

    return results


if __name__ == "__main__":

    results = price_relative_perf_fx_trigger_mc()

    print("Monte Carlo Pricing Results")
    print("-" * 40)

    print(f"Model price:                 ${results['price']:,.2f}")
    print(f"Model price / notional:       {results['price_pct_notional']:.4%}")
    print(f"MC standard error:            ${results['std_error']:,.2f}")
    print(f"MC standard error / notional: {results['std_error_pct_notional']:.4%}")

    print()
    print(f"Undiscounted expected payoff: ${results['undiscounted_expected_payoff']:,.2f}")
    print(f"Discount factor:              {results['discount_factor']:.6f}")

    print()
    print(f"FX trigger probability:       {results['trigger_probability']:.4%}")
    print(f"Equity payoff positive prob:  {results['positive_equity_payoff_probability']:.4%}")
    print(f"Total payoff positive prob:   {results['positive_total_payoff_probability']:.4%}")

    print()
    print(f"Dealer premium:               ${results['dealer_premium']:,.2f}")
    print(f"Dealer premium / notional:    {results['dealer_premium_pct_notional']:.4%}")
    print(f"Model - premium:              ${results['model_minus_premium']:,.2f}")
    print(f"Model - premium / notional:   {results['model_minus_premium_pct_notional']:.4%}")

    print()
    print(f"Correlation eigenvalues:      {results['correlation_eigenvalues']}")