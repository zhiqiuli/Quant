import numpy as np
from scipy.stats import norm

def fx_option_price_from_spot(
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    df_domestic: float,
    df_foreign: float,
    option_type: str = "call",
):
    """
    FX option pricing using forward form,
    but taking spot as input.

    spot: S (domestic per 1 foreign)
    df_domestic: DF_d(T)
    df_foreign:  DF_f(T)
    """

    # compute forward from discount factors
    forward = spot * df_foreign / df_domestic

    if maturity <= 0:
        intrinsic = max(spot - strike, 0.0)
        return intrinsic if option_type == "call" \
            else max(strike - spot, 0.0)

    sqrt_t = np.sqrt(maturity)

    d1 = (np.log(forward / strike) + 0.5 * vol**2 * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type.lower() == "call":
        price = df_domestic * (
            forward * norm.cdf(d1) - strike * norm.cdf(d2)
        )
    elif option_type.lower() == "put":
        price = df_domestic * (
            strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

if __name__ == "__main__":

    T = 90 / 365 # 3m expiry

    r_usd = 0.03671 #
    r_jpy = 0.00614 #

    df_usd = np.exp(-r_usd * T)
    df_jpy = np.exp(-r_jpy * T)

    price = fx_option_price_from_spot(
        spot=155.05,
        strike=153.88,
        maturity=T,
        vol=0.09711,
        df_domestic=df_jpy,  # JPY DF -- domestic currency determines how the payoff function works
        df_foreign=df_usd,  # USD DF
        option_type="call",
    )

    premium_jpy = price * 1_000_000
    print(premium_jpy)