import datetime
import os

import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np

def calcVarCumSum(start_date,
                  calc_date,
                  valuation_date,
                  df_eq_dispersion,
                  iv_bump_size=0,
                  dir=''):

    cumsum = 0

    for i, row in df_eq_dispersion.iterrows():
        ticker = row['Ticker']

        # cumsum log(S_i / S_{i-1})
        variance_return = download_returns_to_csv(ticker,
                                                  start_date,
                                                  calc_date,
                                                  dir=dir)

        # Expected number of observation days
        Expected_N = row['Expected N']

        # Convert back to Timestamp
        start = pd.Timestamp(start_date)
        calc = pd.Timestamp(calc_date)
        end = pd.Timestamp(valuation_date)

        # Use NYSE calendar
        nyse = mcal.get_calendar("XNYS")

        # Realized days (include both endpoints)
        sched_real = nyse.schedule(start_date=start, end_date=calc)
        N_real = len(sched_real)

        # Remaining days (exclude calc_date, include maturity)
        sched_rem = nyse.schedule(start_date=calc + pd.Timedelta(days=1),
                                  end_date=end)

        N_rem = len(sched_rem)

        # Implied vols are used as a proxy here. The more accurate way it so calculate the fwd variance. A separate Python project will cover that part.
        # Implied Vol means Decimal 0.16 = 16%
        # Basket 1
        IV_i = row['20260616 100% ATM VOL As Of 20260227 (BBG)']

        # Basket 2
        # IV_i = row['20270617 100% ATM VOL As Of 20260305 (BBG)']

        # Share Final Realized Vol
        sigma_real_sq  = 252 * variance_return / Expected_N
        sigma_fwd_sq   = ((IV_i + 100 * iv_bump_size) / 100) ** 2

        w_real         = N_real / Expected_N
        w_rem          = N_rem  / Expected_N

        sigma_total_sq = w_real * sigma_real_sq + w_rem * sigma_fwd_sq

        SFRV_i = 100 * np.sqrt(sigma_total_sq)

        # Share Vol Amount $
        SVA_i = row['Volatility Amount (USD)']

        # Share Vol Strike
        SVK_i = row['Volatility Strike']

        # Share Vol Cap
        # Note that, sometime this quantity is 2.5x, then it's 2.5x Volatility Strike
        SVC_i = row['Volatility Cap']

        Val_i = SVA_i * (np.minimum(SFRV_i, SVC_i) - SVK_i)

        if True:
            print(
                f"{ticker}\n"
                f"  N_real={N_real}, N_rem={N_rem}, Expected_N={Expected_N}\n"
                f"  w_real={w_real:.6f}, w_rem={w_rem:.6f}\n"
                f"  sigma_real_sq={sigma_real_sq:.8f}, sigma_fwd_sq={sigma_fwd_sq:.8f}\n"
                f"  sigma_total_sq={sigma_total_sq:.8f}\n"
                f"  SFRV_i(%)={SFRV_i:.4f}, Cap={SVC_i}, Strike={SVK_i}\n"
                f"  SVA_i={SVA_i}, Val_i={Val_i}\n"
            )

        cumsum += Val_i

    return cumsum

def download_returns_to_csv(ticker,
                            start_date,
                            end_date,
                            interval="1d",
                            dir=''):

    # Download data
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No data returned. Check ticker or date range.")

    # Keep only Close column
    df = df[["Close"]].copy()

    # Reset index so Date becomes a column
    df.reset_index(inplace=True)

    # Save to CSV
    filename = f"{ticker}_close_{start_date}_to_{end_date}.csv"
    output_dir = dir + r'/Output/'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir + filename, index=False)

    # print(f"Saved to {filename}")

    prices = df["Close"].values

    variance_return = np.sum(
            np.log(prices[1:] / prices[:-1]) ** 2
    )

    return variance_return


if __name__ == "__main__":

    dir = r'./EqDispersion/'

    # basket 1
    df_eq_dispersion_single_names = pd.read_csv(dir + '/Input/DispersionBasket1-SingleNames.csv')
    df_eq_dispersion_index        = pd.read_csv(dir + '/Input/DispersionBasket1-Index.csv')
    start_date                    = datetime.datetime(2025, 7, 15).strftime("%Y-%m-%d") # +1 day
    calc_date                     = datetime.datetime(2026, 3, 3).strftime("%Y-%m-%d") # +1 day as well, because Yahoo excludes the end_date
    valuation_date                = datetime.datetime(2026, 6, 18).strftime("%Y-%m-%d")

    # basket 2
    # df_eq_dispersion_single_names = pd.read_csv(dir + '/Input/DispersionBasket2-SingleNames.csv')
    # df_eq_dispersion_index        = pd.read_csv(dir + '/Input/DispersionBasket2-Index.csv')
    # start_date                    = datetime.datetime(2026, 2, 28).strftime("%Y-%m-%d") # +1 day
    # calc_date                     = datetime.datetime(2026, 3, 6).strftime("%Y-%m-%d") # +1 day as well, because Yahoo excludes the end_date
    # valuation_date                = datetime.datetime(2027, 6, 17).strftime("%Y-%m-%d")

    single_name_   = calcVarCumSum(start_date,
                                   calc_date,
                                   valuation_date,
                                   df_eq_dispersion_single_names,
                                   iv_bump_size=0,
                                   dir=dir)

    index_         = calcVarCumSum(start_date,
                                   calc_date,
                                   valuation_date,
                                   df_eq_dispersion_index,
                                   iv_bump_size=0,
                                   dir=dir)

    print(f'SingleNames - Index = {single_name_ - index_:,.2f}')

    if False:
        single_name_   = calcVarCumSum(start_date,
                                       calc_date,
                                       valuation_date,
                                       df_eq_dispersion_single_names,
                                       0.01)

        index_         = calcVarCumSum(start_date,
                                       calc_date,
                                       valuation_date,
                                       df_eq_dispersion_index,
                                       0.01)

        print(f'SingleNames - Index = {single_name_ - index_:,.2f}')
