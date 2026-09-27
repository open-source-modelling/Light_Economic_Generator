import numpy as np
import pandas as pd
from term_structure import calculate_instantaneous_forward_rate, calculate_zero_coupon_price

def calculate_vasicek_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_drift: float, sigma: float, gamma: float, tolerance: float)->dict:
    """
    Simulates a series of stochastic interest rate paths using the Vasicek model

        dr(t) = gamma (mu - r(t)) dt + sigma dW(t)

    with the exact transition of the short rate over each time step:

        r(t_i) = r(t_i-1) exp(-gamma dt) + mu (1-exp(-gamma dt))
                 + sigma sqrt((1-exp(-2 gamma dt))/(2 gamma)) Z_i

    The short rate is split into a deterministic and a stochastic part,
    r(t) = m(t) + x(t), where

        m(t) = mu + (r(0) - mu) exp(-gamma t)
        dx(t) = -gamma x(t) dt + sigma dW(t),   x(0) = 0.

    The pair (x(t_i), integral of x over [t_i-1, t_i]) is normally distributed
    given x(t_i-1), and is sampled exactly. The discount factor is

        D(t) = exp(-integral of m over [0, t] - integral of x over [0, t])

    so the scheme has no discretisation error for any time step.

    Only the initial short rate r(0) is taken from the term structure. The
    simulated discount factors therefore do not reproduce the input term structure.

    Args:
        num_paths (int): number of paths to simulate.
        num_steps (int): number of time steps per path.
        end_time (int): end of the modelling window (in years).
            (Ex. a modelling window of 50 years means T=50).
        function_zero_coupon_price (function): function that calculates the price of a
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        mean_drift (float): long-term mean parameter mu of the Vasicek model.
        sigma (float): volatility parameter sigma of the Vasicek model.
        gamma (float): mean reversion speed parameter gamma of the Vasicek model.
        tolerance (float): size of the increment used for finite
            difference approximation of the initial short rate.

    Returns:
        dict: A dictionary containing arrays with time steps, interest rate paths,
            discount factors and bank account values.
            time (array): array of time steps.
            R (array): array of interest rate paths with
              shape (num_paths, num_steps+1).
            M (array): array of discount factors with
              shape (num_paths, num_steps+1).
            I (array): array of bank account values (1/M) with
              shape (num_paths, num_steps+1).

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.

    Exact scheme: P. Glasserman (2003), Monte Carlo Methods in Financial Engineering, 
    Springer, section 3.3 "Gaussian Short Rate Models".
    """
    if num_paths < 1:
        raise ValueError("Number of paths must be at least 1")
    if num_steps < 1:
        raise ValueError("Number of steps must be at least 1")
    if end_time <= 0:
        raise ValueError("End time T must be positive")
    if gamma == 0:
        raise ValueError("Mean reversion speed gamma must not be 0. The limit gamma = 0 (Brownian motion) is not supported")
    if gamma < 0:
        raise ValueError("Mean reversion speed gamma must be positive")
    if sigma < 0:
        raise ValueError("Volatility sigma must not be negative")
    if tolerance <= 0:
        raise ValueError("Tolerance epsilon must be positive")

    # Initial instantaneous forward rate at time t-> 0 (also spot rate at time 0).
    # r(0) = f(0,0) = - partial derivative of log(P_mkt(0, tolerance) w.r.t tolerance)
    r0 = calculate_instantaneous_forward_rate(tolerance, function_zero_coupon_price, tolerance)

    # Generate two independent sources of random noise. The first drives the short 
    # rate, the second the part of its integral that is independent of the short rate.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
    Z2 = np.random.normal(0.0, 1.0, [num_paths, num_steps])

    # Vector of time moments.
    time = np.linspace(0, end_time, num_steps+1)
    dt = end_time/float(num_steps) # Size of increments between two steps

    # Stochastic part x(t) of the short rate and its integral over [0, t].
    x = np.zeros([num_paths, num_steps+1])
    integral_x = np.zeros([num_paths, num_steps+1])

    # Constants of the exact transition over one time step.
    decay = np.exp(-gamma * dt)                                       # Decay of the previous value of x
    sd_term = np.sqrt(sigma**2/(2*gamma)*(1-np.exp(-2*gamma*dt)))     # Standard deviation of the shock to x

    # Variance of the integral of x over one time step and its covariance with x,
    # divided by sigma^2.
    var_integral = (dt - 2.0*(1.0-decay)/gamma + (1.0-np.exp(-2.0*gamma*dt))/(2.0*gamma))/gamma**2
    cov_x_integral = (1.0-decay)**2/(2.0*gamma**2)

    # Cholesky decomposition of the covariance matrix.
    loading_x = np.sqrt((1-np.exp(-2*gamma*dt))/(2*gamma))
    loading_integral_1 = cov_x_integral/loading_x
    loading_integral_2 = np.sqrt(max(var_integral - loading_integral_1**2, 0.0))

    for iTime in range(1, num_steps+1): # For each time increment
        # Making sure the samples from the normal distribution have a mean of 0
        # and variance 1
        if num_paths > 1:
            Z[:, iTime-1] = (Z[:, iTime-1]-np.mean(Z[:, iTime-1]))/np.std(Z[:, iTime-1])
            Z2[:, iTime-1] = (Z2[:, iTime-1]-np.mean(Z2[:, iTime-1]))/np.std(Z2[:, iTime-1])

        # Apply the exact transition of x and of its integral at each time increment.
        x_previous = x[:, iTime-1]
        x[:, iTime] = x_previous*decay + sd_term*Z[:, iTime-1]
        integral_step = x_previous*(1.0-decay)/gamma + sigma*(loading_integral_1*Z[:, iTime-1] + loading_integral_2*Z2[:, iTime-1])
        integral_x[:, iTime] = integral_x[:, iTime-1] + integral_step

    # Deterministic part of the short rate and its integral.
    deterministic_rate = mean_drift + (r0 - mean_drift)*np.exp(-gamma*time)
    integral_deterministic_rate = mean_drift*time + (r0 - mean_drift)*(1-np.exp(-gamma*time))/gamma

    # Short rate. The first interest rate equals the instantaneous forward (spot)
    # rate at time 0.
    R = deterministic_rate + x

    # Discount factor D(t) = exp(-integral of the short rate over [0, t]).
    M = np.exp(-integral_deterministic_rate - integral_x)
    I = 1/M
    # Output is a dictionary with time moments, the interest rate paths, the discount
    # factors and the bank account values.
    paths = {"time":time, "R":R, "M":M, "I":I}
    return paths


def vasicek_main_calculation(num_paths: int, num_steps: int, end_time: int, mean_drift: float, sigma: float, gamma: float, function_zero_coupon_price: callable, tolerance: float)-> list:
    """
    Simulates the Vasicek model and calculates the average discount factor, 
    which is the Monte Carlo price of a zero-coupon bond (ZCB).
    
    Args:
        num_paths (int): number of Monte Carlo simulation paths.
        num_steps (int): number of time steps per path.
        end_time (int): length in years of the modelling window (Ex. 50 years means t=50).
        mean_drift (float): long-term mean parameter mu of the Vasicek model.
        sigma (float): volatility parameter sigma of the Vasicek model.
        gamma (float): mean reversion speed parameter gamma of the Vasicek model.
        function_zero_coupon_price (function): function that calculates the price of a zero coupon bond issued. 
           at time 0 that matures at time t, with a notional amount 1 and discounted using
           the assumed term structure.
        tolerance (float): the size of the increment  used for finite difference approximation.
    
    Returns:
        t : time increments.
        P : average of the simulated discount factors.
        implied_term_structure : term structure provided as input into the V simulation.
        M : discount factors.
        I : bank account values.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.        
    """
 
    paths = calculate_vasicek_paths(num_paths, num_steps, end_time, function_zero_coupon_price, mean_drift, sigma, gamma, tolerance)
    M = paths["M"]
    t = paths["time"]
    I = paths["I"]
    implied_term_structure = function_zero_coupon_price(t)
    # Monte Carlo price of a ZCB for every time step
    P = np.zeros([num_steps+1])
    for i in range(0, num_steps+1):
        P[i] = np.mean(M[:, i])
    
    return [t, P, implied_term_structure, M, I]


def set_up_vasicek(asset_id: int, modeling_parameters: dict, zero_coupon_price: callable) -> pd.DataFrame:


    num_paths = modeling_parameters["num_paths"]  # Number of stochastic scenarios
    num_steps = modeling_parameters["num_steps"]  # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = modeling_parameters["end_time"]    # Time horizon in years (A time horizon of 50 years; T=50)
    mu =  modeling_parameters["mu"]               # Vasicek long term mean parameter mu
    gamma = modeling_parameters["gamma"]          # Vasicek mean reversion speed parameter gamma
    sigma = modeling_parameters["sigma"]          # Vasicek volatility parameter sigma
    tolerance =  modeling_parameters["tolerance"] # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)
    type = modeling_parameters["curve_type"]

    # Final comparison
    [t, P, implied_term_structure, M, I] = vasicek_main_calculation(num_paths, num_steps, end_time, mu, sigma, gamma, zero_coupon_price, tolerance)

    run_name = "V-"+str(asset_id)

    if type=="I":
        outTmp = I
    elif type=="D":
        outTmp = M
    else:
        raise ValueError

    multi_index_list = []
    for scenario in list(range(0,num_paths)):
        multi_index_list.append((run_name,scenario))

    multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=('Run', 'Scenario_number'))
    scenarios = pd.DataFrame(data = outTmp, columns=t, index=multi_index)

    return scenarios


