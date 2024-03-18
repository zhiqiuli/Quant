import math
import numpy as np
from scipy.stats import norm

def american_call_option_simulation(S, K, r, sigma, T, n_steps, n_simulations, option_type):
    """
    Monte Carlo simulation for pricing American call options.

    Parameters:
        S: float
            Current stock price.
        K: float
            Strike price.
        r: float
            Risk-free rate.
        sigma: float
            Volatility of the underlying stock.
        T: float
            Time to expiration.
        n_steps: int
            Number of time steps for simulation.
        n_simulations: int
            Number of simulations to run.

    Returns:
        float
            Option price.
    """
    dt = T / n_steps
    option_price = 0
    
    rand_nums = np.random.normal(0, 1, [n_simulations, n_steps])
    option_prices = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        stock_price = S
        for j in range(n_steps):
            # Update stock price using geometric Brownian motion
            stock_price *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_nums[i][j])
            # Calculate intrinsic value of the option
            if option_type == 'call':
                intrinsic_value = max(stock_price - K, 0)
            elif option_type == 'put':
                intrinsic_value = max(K - stock_price, 0)
            else:
                raise ValueError
            # Calculate option price at this time step
            option_price = max(option_prices[i], intrinsic_value)  # Update option price if intrinsic value is higher
            # Calculate expected value of continuing to hold the option
            expected_value = np.exp(-r * (j + 1) * dt) * option_price
            # Update option price if expected value is higher
            option_prices[i] = max(option_prices[i], expected_value)            
    # Average option price over all simulations
    option_price = np.mean(option_prices)
    return option_price



def european_option_simulation(S, K, r, sigma, T, n_simulations, option_type):
    """
    Monte Carlo simulation for pricing European options.

    Parameters:
        S: float
            Current stock price.
        K: float
            Strike price.
        r: float
            Risk-free rate.
        sigma: float
            Volatility of the underlying stock.
        T: float
            Time to expiration.
        n_simulations: int
            Number of simulations to run.

    Returns:
        float
            Option price.
    """
    dt = T
    rand = np.random.normal(0, 1, n_simulations)
    # Simulating stock prices at expiration
    ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)
    # Calculating option payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError
    # Discounting the expected payoffs
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price



def black_scholes(S, K, T, r, sigma, option_type):
  """Calculates the Black-Scholes price of an option.

  Args:
    S: The underlying asset price.
    K: The strike price.
    T: The time to expiration in years.
    r: The risk-free interest rate.
    sigma: The volatility of the underlying asset.
    option_type: A string representing the option type, "call" or "put".

  Returns:
    The Black-Scholes price of the option.
  """

  d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
  d2 = d1 - sigma * math.sqrt(T)

  if option_type == "call":
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
  elif option_type == "put":
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
  else:
    raise ValueError("Invalid option type: {}".format(option_type))



def binomial_option_pricing(S, K, T, r, sigma, n, option_type):
    delta_t = T / n
    '''
    The up factor in a binomial tree model is determined based on the volatility of
    the underlying asset and the length of each time step. The standard approach to
    determining the up factor is through the concept of "geometric Brownian motion,"
    which assumes that the logarithm of the asset price follows a normal distribution
    with a mean and standard deviation.
    '''
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    '''
    The risk-free interest rate (r) comes into play when calculating the probability
    of upward (p) and downward (1−p) movements in the binomial tree.
    '''
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    stock_tree = np.zeros((n + 1, n + 1))
    option_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S
    
    '''
    build up the underlying asset price tree
    '''
    for i in range(1, n + 1):
        stock_tree[i, 0] = stock_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_tree[i, j] = stock_tree[i - 1, j - 1] * d
        
    if option_type == "call":
        option_tree[-1, :] = np.maximum(0, stock_tree[-1, :] - K)
    elif option_type == "put":
        option_tree[-1, :] = np.maximum(0, K - stock_tree[-1, :])
    else:
        raise ValueError("Invalid option type. Please choose 'call' or 'put'.")
    
    '''
    backward expectation step by step
    '''
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_tree[i, j] = np.exp(-r * delta_t) * ((1-p) * option_tree[i + 1, j + 1] +  p * option_tree[i + 1, j])

    return option_tree[0, 0], stock_tree, option_tree



def binomial_american_option_pricing (S, K, T, r, sigma, n, option_type):
    delta_t = T / n
    '''
    The up factor in a binomial tree model is determined based on the volatility of
    the underlying asset and the length of each time step. The standard approach to
    determining the up factor is through the concept of "geometric Brownian motion,"
    which assumes that the logarithm of the asset price follows a normal distribution
    with a mean and standard deviation.
    '''
    u = np.exp(sigma * np.sqrt(delta_t))
    d = 1 / u
    '''
    The risk-free interest rate (r) comes into play when calculating the probability
    of upward (p) and downward (1−p) movements in the binomial tree.
    '''
    p = (np.exp(r * delta_t) - d) / (u - d)
    
    stock_tree = np.zeros((n + 1, n + 1))
    option_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S
    
    '''
    build up the underlying asset price tree
    '''
    for i in range(1, n + 1):
        stock_tree[i, 0] = stock_tree[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_tree[i, j] = stock_tree[i - 1, j - 1] * d
        
    if option_type == "call":
        option_tree[-1, :] = np.maximum(0, stock_tree[-1, :] - K)
    elif option_type == "put":
        option_tree[-1, :] = np.maximum(0, K - stock_tree[-1, :])
    else:
        raise ValueError("Invalid option type. Please choose 'call' or 'put'.")
    
    '''
    backward expectation step by step
    '''
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            if option_type == "call":
                option_tree[i, j] = max(stock_tree[i, j] - K, np.exp(-r * delta_t) * ((1-p) * option_tree[i + 1, j + 1] +  p * option_tree[i + 1, j]))
            elif option_type == "put":
                option_tree[i, j] = max(K - stock_tree[i, j], np.exp(-r * delta_t) * ((1-p) * option_tree[i + 1, j + 1] +  p * option_tree[i + 1, j]))
            else:
                raise ValueError
    
    return option_tree[0, 0], stock_tree, option_tree

# Example usage:

S = 100
K = 105
T = 1
r = 0.05
sigma = 0.2
option_type = 'call'
n_steps = 5            # for american/sim only -- a larger n_steps means more chances of exercise
n_simulations = 100000 # for both european and american
n_trees = 100          # for binomial tree

option_price = black_scholes (S, K, T, r, sigma, option_type)
print(f"European {option_type} option price (analytic): {option_price}")

option_price = european_option_simulation (S, K, r, sigma, T, n_simulations, option_type)
print(f"European {option_type} option price (sim): {option_price}")

option_price, _, _ = binomial_option_pricing (S, K, T, r, sigma, n_trees, option_type)
print(f"European {option_type} option price (tree): {option_price}")

#option_price = american_call_option_simulation (S, K, r, sigma, T, n_steps, n_simulations, option_type)
#print(f"American {option_type} option price (sim):", option_price)

option_price, _, _ = binomial_american_option_pricing (S, K, T, r, sigma, n_trees, option_type)
print(f"American {option_type} option price (tree):", option_price)

