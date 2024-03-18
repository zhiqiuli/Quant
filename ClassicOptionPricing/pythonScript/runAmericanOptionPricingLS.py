"""Test usage demo including imports"""
from longstaff_schwartz.algorithm import longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion
import numpy as np

# Model parameters
t = np.linspace(0, 1, 101) # timegrid for simulation
r = 0.05 # riskless rate
sigma = 0.2 # annual volatility of underlying
n = 100000 # number of simulated paths

# Simulate the underlying
gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
rnd = np.random.RandomState(1234)
x = gbm.simulate(t, n, rnd)  # x.shape == (t.size, n)

# Payoff (exercise) function
strike = 1.05

option_type = 'call'

if option_type == 'put':
    def payoff_func(spot):
        return np.maximum(strike - spot, 0.0)
elif option_type == 'call':
    def payoff_func(spot):
        return np.maximum(spot - strike, 0.0)
else:
    raise ValueError

# Discount factor function
def constant_rate_df(t_from, t_to):
    return np.exp(-r * (t_to - t_from))

# Approximation of continuation value
def fit_quadratic(x, y):
    return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)

# Selection of paths to consider for exercise
# (and continuation value approxmation)
def itm(payoff, spot):
    return payoff > 0

# Run valuation of American put option
npv_american = longstaff_schwartz(
    x, t, constant_rate_df, fit_quadratic, payoff_func, itm
)

# European put option for comparison
# npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()

# Check results
print(f"American {option_type} option price (sim):", npv_american * 100)
#print(npv_european)
