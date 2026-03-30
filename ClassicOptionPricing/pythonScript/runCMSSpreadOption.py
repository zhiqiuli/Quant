import math
from scipy.stats import norm

def price_cms_spread_option(
    N,
    T,
    K,
    F_a,
    F_b,
    conv_a,
    conv_b,
    sigma_a,
    sigma_b,
    rho,
    discount_factor=1.0,
    option_type="cap"
):
    """
    Price a CMS spread option using Bachelier approximation.

    Parameters
    ----------
    N : float
        Notional
    T : float
        Time to maturity (years)
    K : float
        Strike (decimal, e.g. 0.005 for 0.5%)
    F_a : float
        Forward swap rate for CMS_a (e.g. 10Y)
    F_b : float
        Forward swap rate for CMS_b (e.g. 2Y)
    conv_a : float
        Convexity adjustment for CMS_a
    conv_b : float
        Convexity adjustment for CMS_b
    sigma_a : float
        Vol of swap rate a (from swaption, e.g. 2Yx10Y)
    sigma_b : float
        Vol of swap rate b (e.g. 2Yx2Y)
    rho : float
        Correlation between swap rates
    discount_factor : float
        Discount factor to maturity (default = 1)

    Returns
    -------
    dict with:
        price, spread, spread_vol, d
    """

    # -----------------------------
    # 1. CMS rates
    # -----------------------------
    CMS_a = F_a + conv_a
    CMS_b = F_b + conv_b

    S0 = CMS_a - CMS_b

    # -----------------------------
    # 2. Spread volatility
    # -----------------------------
    sigma_S_sq = (
        sigma_a**2
        + sigma_b**2
        - 2 * rho * sigma_a * sigma_b
    )
    sigma_S = math.sqrt(max(sigma_S_sq, 0.0))  # numerical safety

    # -----------------------------
    # 3. Bachelier pricing
    # -----------------------------
    sigma_T = sigma_S * math.sqrt(T)

    if sigma_T < 1e-12:
        if option_type == "cap":
            payoff = max(S0 - K, 0.0)
        elif option_type == "floor":
            payoff = max(K - S0, 0.0)
        else:
            raise ValueError("option_type must be 'cap' or 'floor'")

        price = discount_factor * N * payoff
        d = float("inf") if payoff > 0 else -float("inf")

    else:
        d = (S0 - K) / sigma_T

        if option_type == "cap":
            price_per_unit = (
                (S0 - K) * norm.cdf(d)
                + sigma_T * norm.pdf(d)
            )

        elif option_type == "floor":
            price_per_unit = (
                (K - S0) * norm.cdf(-d)
                + sigma_T * norm.pdf(d)
            )

        else:
            raise ValueError("option_type must be 'cap' or 'floor'")

        price = discount_factor * N * price_per_unit

    return {
        "price": price,
        "spread": S0,
        "spread_vol": sigma_S,
        "d": d,
    }

def compute_cms_spread_risks(
        bump_rate=1e-4,  # 1 bp
        bump_vol=0.01,  # 1%
        bump_corr=0.01,
        bump_spread=1e-4,
        **kwargs
):
    base = price_cms_spread_option(**kwargs)["price"]

    '''
    # Copy original inputs
    new_inputs = kwargs.copy()
    
    # Bump F_a
    new_inputs["F_a"] = kwargs["F_a"] + bump_rate
    
    # Reprice
    result = price_cms_spread_option(**new_inputs)
    
    # Extract price
    up = result["price"]
    '''

    # -----------------------------
    # DV01 (tenor a)
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "F_a": kwargs["F_a"] + bump_rate}
    )["price"]
    dv01_a = up - base

    # -----------------------------
    # DV01 (tenor b)
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "F_b": kwargs["F_b"] + bump_rate}
    )["price"]
    dv01_b = up - base

    # -----------------------------
    # Vega (tenor a)
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "sigma_a": kwargs["sigma_a"] + bump_vol}
    )["price"]
    vega_a = up - base

    # -----------------------------
    # Vega (tenor b)
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "sigma_b": kwargs["sigma_b"] + bump_vol}
    )["price"]
    vega_b = up - base

    # -----------------------------
    # Corr01
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "rho": kwargs["rho"] + bump_corr}
    )["price"]
    corr01 = up - base

    # -----------------------------
    # Spread01 (direct bump)
    # -----------------------------
    up = price_cms_spread_option(
        **{
            **kwargs,
            "conv_a": kwargs["conv_a"] + bump_spread / 2,
            "conv_b": kwargs["conv_b"] - bump_spread / 2,
        }
    )["price"]
    spread01 = up - base

    # -----------------------------
    # Convexity risk
    # -----------------------------
    up = price_cms_spread_option(
        **{**kwargs, "conv_a": kwargs["conv_a"] + bump_rate}
    )["price"]
    conv_a_risk = up - base

    up = price_cms_spread_option(
        **{**kwargs, "conv_b": kwargs["conv_b"] + bump_rate}
    )["price"]
    conv_b_risk = up - base

    return {
        "DV01_10Y": dv01_a,
        "DV01_2Y": dv01_b,
        "Vega_10Y": vega_a,
        "Vega_2Y": vega_b,
        "Corr01": corr01,
        "Spread01": spread01,
        "Convexity_10Y": conv_a_risk,
        "Convexity_2Y": conv_b_risk,
    }

if __name__ == "__main__":

    N = 1_000_000
    T = 10.0
    K = 0.005
    F_a = 0.0380
    F_b = 0.0300
    conv_a = 0.0020
    conv_b = 0.0003
    sigma_a = 0.22
    sigma_b = 0.18
    rho = 0.90

    cap = price_cms_spread_option(
        N, T, K, F_a, F_b, conv_a, conv_b, sigma_a, sigma_b, rho,
        option_type="cap")["price"]

    floor = price_cms_spread_option(
        N, T, K, F_a, F_b, conv_a, conv_b, sigma_a, sigma_b, rho,
        option_type="floor")["price"]

    S0 = (F_a + conv_a) - (F_b + conv_b)
    print(cap - floor, "vs", (S0 - K) * N)

    '''
    Because now Python understands:
    
    kwargs = {
        "N": N,
        "T": T,
        "K": K,
        ...
    }
    '''
    risks = compute_cms_spread_risks(
        N=N,
        T=T,
        K=K,
        F_a=F_a,
        F_b=F_b,
        conv_a=conv_a,
        conv_b=conv_b,
        sigma_a=sigma_a,
        sigma_b=sigma_b,
        rho=rho,
        option_type="cap"
    )

    for k, v in risks.items():
        print(f"{k}: {v:,.2f}")