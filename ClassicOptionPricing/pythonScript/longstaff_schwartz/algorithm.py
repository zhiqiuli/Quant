import numpy as np
from numpy.polynomial import Polynomial


def longstaff_schwartz_iter(X, t, df, fit, exercise_payoff, itm_select=None):
    # given no prior exercise we just receive the final payoff
    cashflow = exercise_payoff(X[-1, :])
    # iterating backwards in time
    for i in reversed(range(1, X.shape[0] - 1)):
        # discount cashflows from next period
        cashflow = cashflow * df(t[i], t[i + 1])
        x = X[i, :]
        # exercise value for time t[i]
        exercise = exercise_payoff(x)
        # boolean index of all in-the-money paths
        # (paths considered for exercise)
        itm = itm_select(exercise, x) if itm_select else np.full(x.shape, True)
        # fit curve
        fitted = fit(x[itm], cashflow[itm])
        # approximate continuation value
        continuation = fitted(x)
        # boolean index where exercise is beneficial
        ex_idx = itm & (exercise > continuation)
        # update cashflows with early exercises
        cashflow[ex_idx] = exercise[ex_idx]

        yield cashflow, x, fitted, continuation, exercise, ex_idx


def longstaff_schwartz(X, t, df, fit, exercise_payoff, itm_select=None):
    for cashflow, *_ in longstaff_schwartz_iter(
        X, t, df, fit, exercise_payoff, itm_select
    ):
        pass
    return cashflow.mean(axis=0) * df(t[0], t[1])


def ls_american_option_quadratic_iter(X, t, r, strike):
    # given no prior exercise we just receive the payoff of a European option
    cashflow = np.maximum(strike - X[-1, :], 0.0)
    # iterating backwards in time
    for i in reversed(range(1, X.shape[0] - 1)):
        # discount factor between t[i] and t[i+1]
        df = np.exp(-r * (t[i + 1] - t[i]))
        # discount cashflows from next period
        cashflow = cashflow * df
        x = X[i, :]
        # exercise value for time t[i]
        exercise = np.maximum(strike - x, 0.0)
        # boolean index of all in-the-money paths
        itm = exercise > 0
        # fit polynomial of degree 2
        fitted = Polynomial.fit(x[itm], cashflow[itm], 2)
        # print(fitted)
        # approximate continuation value
        continuation = fitted(x)
        # boolean index where exercise is beneficial
        ex_idx = itm & (exercise > continuation)
        # update cashflows with early exercises
        cashflow[ex_idx] = exercise[ex_idx]

        yield cashflow, x, fitted, continuation, exercise, ex_idx


def longstaff_schwartz_american_option_quadratic(X, t, r, strike):
    for cashflow, *_ in ls_american_option_quadratic_iter(X, t, r, strike):
        pass
    return cashflow.mean(axis=0) * np.exp(-r * (t[1] - t[0]))


if __name__ == '__main__':
    
    # Longstaff-Schwatz paper's results
    
    X = np.array(
        [
            [1.00, 1.09, 1.08, 1.34],
            [1.00, 1.16, 1.26, 1.54],
            [1.00, 1.22, 1.07, 1.03],
            [1.00, 0.93, 0.97, 0.92],
            [1.00, 1.11, 1.56, 1.52],
            [1.00, 0.76, 0.77, 0.90],
            [1.00, 0.92, 0.84, 1.01],
            [1.00, 0.88, 1.22, 1.34],
        ]
    ).T
    t = np.array([0, 1, 2, 3])
    r = 0.06
    strike = 1.1
    coef2 = np.array([-1.070, 2.983, -1.814])  # -1.813 in paper
    coef1 = np.array([2.038, -3.335, 1.356])
    
    npv_american = longstaff_schwartz_american_option_quadratic(X, t, r, strike)
    print(round(npv_american,4)) # .1144
    
    npv_european = np.mean(np.exp(-t[-1] * r) * np.maximum(0, strike - X[-1,:]))
    print(round(npv_european,4)) # .0564
    
    intermediate = list(ls_american_option_quadratic_iter(X, t, r, strike))
    cashflow, x, fitted, continuation, exercise, ex_idx = intermediate[0]
    fitted_coef2 = np.round(fitted.convert(domain=[-1, 1]).coef, 3)
    print(coef2, fitted_coef2)
    cashflow, x, fitted, continuation, exercise, ex_idx = intermediate[1]
    fitted_coef1 = np.round(fitted.convert(domain=[-1, 1]).coef, 3)
    print(coef1, fitted_coef1)
    
    