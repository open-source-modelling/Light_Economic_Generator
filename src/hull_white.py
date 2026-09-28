import numpy as np
import pandas as pd
from term_structure import calculate_instantaneous_forward_rate, calculate_zero_coupon_price


def calculate_hull_white_theta(mean_reversion_rate: float, volatility: float, function_zero_coupon_price: callable, tolerance: float) -> callable:
    """
    Calculates the theta value for the Hull-White model 
    using a numeric approximation of the instantaneous forward rate 
    and the spot rate.

    Args:
        mean_reversion_rate (float): Mean reversion rate parameter a.
        volatility (float): Volatility parameter sigma.
        function_zero_coupon_price (function handle): Function that calculates the price of a
            zero-coupon bond as a function of time.
        tolerance (float): Increment of time used in the numeric calculation of the 
            derivative of the instantaneous forward rate.

    Returns:
        theta (function): Function that returns the parameter theta of 
            Hull-White model at the time t implied by the calibration.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.
    """
    if mean_reversion_rate == 0:
        raise ValueError("Mean reversion rate a must not be 0. The limit a = 0 (Ho-Lee model) is not supported")
    if mean_reversion_rate < 0:
        raise ValueError("Mean reversion rate a must be positive")

    def theta(t:float)->float:
        insta_forward_term = (calculate_instantaneous_forward_rate(t+tolerance, function_zero_coupon_price, tolerance) 
                                         -calculate_instantaneous_forward_rate(t-tolerance,function_zero_coupon_price,tolerance))/(2.0*tolerance)
                                         
        forward_term = mean_reversion_rate*calculate_instantaneous_forward_rate(t, function_zero_coupon_price, tolerance)
        variance_term = volatility**2/(2.0*mean_reversion_rate)*(1.0-np.exp(-2.0*mean_reversion_rate*t))
        return insta_forward_term + forward_term + variance_term
    return theta

def calculate_hull_white_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_reversion_rate: float, volatility: float, tolerance: float, rng: int | np.random.Generator | None = None)->dict:
    """
    Simulates a series of stochastic interest rate paths using the Hull-White model

        dr(t) = (theta(t) - a r(t)) dt + sigma dW(t)

    with the exact simulation scheme. The short rate is split into a deterministic 
    and a stochastic part, r(t) = alpha(t) + x(t), where

        alpha(t) = f(0,t) + sigma^2/(2 a^2) (1-exp(-a t))^2
        dx(t) = -a x(t) dt + sigma dW(t),   x(0) = 0.

    The pair (x(t_i), integral of x over [t_i-1, t_i]) is normally distributed 
    given x(t_i-1), and is sampled exactly. The discount factor is 

        D(t) = P(0,t) exp(-V(t)/2 - integral of x over [0, t])

    where V(t) is the variance of the integral of x. The scheme has no 
    discretisation error for any time step, and E[D(t)] = P(0,t).

    Args:
        num_paths (int): number of paths to simulate.
        num_steps (int): number of time steps per path.
        end_time (float): end of the modelling window (in years). 
            (Ex. a modelling window of 50 years means T=50).
        function_zero_coupon_price (function): function that calculates the price of a 
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        mean_reversion_rate (float): mean reversion speed parameter a of 
            the Hull-White model.
        volatility (float): volatility parameter sigma of the Hull-White model.
        tolerance (float): size of the increment used for finite
            difference approximation of the instantaneous forward rate.
        rng (int, Generator or None): seed or numpy random number generator used to
            generate the paths. The same seed gives the same paths. If None, the global
            numpy random state is used (set with np.random.seed).

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
    Springer, section 3.3 "Gaussian Short Rate Models". See also V. Ostrovski (2013), 
    Efficient and Exact Simulation of the Hull-White Model, SSRN 2304848.
    """
    if num_paths < 1:
        raise ValueError("Number of paths must be at least 1")
    if num_steps < 1:
        raise ValueError("Number of steps must be at least 1")
    if end_time <= 0:
        raise ValueError("End time T must be positive")
    if mean_reversion_rate == 0:
        raise ValueError("Mean reversion rate a must not be 0. The limit a = 0 (Ho-Lee model) is not supported")
    if mean_reversion_rate < 0:
        raise ValueError("Mean reversion rate a must be positive")
    if volatility < 0:
        raise ValueError("Volatility sigma must not be negative")
    if tolerance <= 0:
        raise ValueError("Tolerance epsilon must be positive")

    a = mean_reversion_rate

    # Vector of time moments.
    time = np.linspace(0, end_time, num_steps+1) 
    dt = end_time/float(num_steps) # Size of increments between two steps

    # Deterministic part of the short rate:
    # alpha(t) = f(0,t) + sigma^2/(2 a^2) (1-exp(-a t))^2.
    forward_rate = calculate_instantaneous_forward_rate(time, function_zero_coupon_price, tolerance)
    alpha = forward_rate + volatility**2/(2.0*a**2)*(1.0-np.exp(-a*time))**2

    # Moments of x(t_i) and of the integral of x over one time step, given x(t_i-1), 
    # divided by sigma^2.
    decay = np.exp(-a*dt)
    var_x = (1.0-np.exp(-2.0*a*dt))/(2.0*a)
    var_integral = (dt - 2.0*(1.0-decay)/a + (1.0-np.exp(-2.0*a*dt))/(2.0*a))/a**2
    cov_x_integral = (1.0-decay)**2/(2.0*a**2)

    # Cholesky decomposition of the covariance matrix.
    loading_x = np.sqrt(var_x)
    loading_integral_1 = cov_x_integral/loading_x
    loading_integral_2 = np.sqrt(max(var_integral - loading_integral_1**2, 0.0))

    # Generate two independent sources of random noise. Without a seed or generator,
    # the global numpy random state is used.
    random_source = np.random if rng is None else np.random.default_rng(rng)
    Z1 = random_source.normal(0.0, 1.0, [num_paths, num_steps])
    Z2 = random_source.normal(0.0, 1.0, [num_paths, num_steps])

    # Making sure the samples from the normal distribution have a mean of 0 
    # and variance 1 at each time increment.
    if num_paths > 1:
        Z1 = (Z1 - np.mean(Z1, axis=0)) / np.std(Z1, axis=0)
        Z2 = (Z2 - np.mean(Z2, axis=0)) / np.std(Z2, axis=0)

    x = np.zeros([num_paths, num_steps+1])
    integral_x = np.zeros([num_paths, num_steps+1])

    for iTime in range(1, num_steps+1): # For each time increment
        x_previous = x[:, iTime-1]
        x[:, iTime] = x_previous*decay + volatility*loading_x*Z1[:, iTime-1]
        integral_step = x_previous*(1.0-decay)/a + volatility*(loading_integral_1*Z1[:, iTime-1] + loading_integral_2*Z2[:, iTime-1])
        integral_x[:, iTime] = integral_x[:, iTime-1] + integral_step

    R = alpha + x

    # Variance of the integral of x over [0, t].
    variance_integral = volatility**2/a**2*(time - 2.0*(1.0-np.exp(-a*time))/a + (1.0-np.exp(-2.0*a*time))/(2.0*a))

    # Discount factor D(t) = P(0,t) exp(-V(t)/2 - integral of x).
    M = function_zero_coupon_price(time)*np.exp(-0.5*variance_integral - integral_x)
    I = 1/M
    # Output is a dictionary with time moments, the interest rate paths, the discount
    # factors and the bank account values.
    paths = {"time":time, "R":R, "M":M, "I":I}
    return paths


def hull_white_main_calculation(num_paths: int, num_steps: int, end_time: int, mean_reversion_rate: float, volatility:float, function_zero_coupon_price: callable, tolerance: float, rng: int | np.random.Generator | None = None):
    """
    Calculates and plots the prices of zero-coupon bonds (ZCB) calculated 
    using the Hull-White model`s analytical formula and the Monte Carlo simulation.
    
    Args:
        num_paths (int): number of Monte Carlo simulation paths.
        num_steps (int): number of time steps per path.
        end_time (int): length in years of the modelling window (Ex. 50 years means t=50).
        mean_reversion_rate (float): mean reversion rate parameter a of the Hull-White model.
        volatility (float): volatility parameter sigma of the Hull-White model.
        function_zero_coupon_price (function): function that calculates the price of a zero coupon bond issued. 
           at time 0 that matures at time t, with a notional amount 1 and discounted using
           the assumed term structure.
        tolerance (float): the size of the increment  used for finite difference approximation.
        rng (int, Generator or None): seed or numpy random number generator. If None,
            the global numpy random state is used (set with np.random.seed).

    Returns:
        t : time increments.
        P : average of the sumulated paths.
        implied_term_structure : term structure provided as input into the HW simulation.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.
    """
 
    paths = calculate_hull_white_paths(num_paths, num_steps, end_time, function_zero_coupon_price, mean_reversion_rate, volatility, tolerance, rng)
    M = paths["M"]
    t = paths["time"]
    I = paths["I"]
    implied_term_structure = function_zero_coupon_price(t)
    # Compare the price of an option on a ZCB from Monte Carlo and the analytical expression
    P = np.zeros([num_steps+1])
    for i in range(0, num_steps+1):
        P[i] = np.mean(M[:, i])
    

    return [t, P, implied_term_structure, M, I]


def set_up_hull_white(asset_id: int, modeling_parameters: dict, zero_coupon_price: callable)->pd.DataFrame:

    num_paths = modeling_parameters["num_paths"] # Number of stochastic scenarios
    num_steps = modeling_parameters["num_steps"] # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = modeling_parameters["end_time"]  # Time horizon in years (A time horizon of 50 years; T=50)
    a =  modeling_parameters["a"]        # Hull-White mean reversion parameter a
    sigma = modeling_parameters["sigma"]    # Hull-White volatility parameter sigma
    tolerance =  modeling_parameters["tolerance"] # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)
    type = modeling_parameters["curve_type"]
    seed = modeling_parameters.get("seed") # Seed of the random number generator (None: different scenarios in every run)
    if type not in ["I", "D"]:
        raise ValueError(f"Output type must be I (index) or D (discount factor), not {type!r}")

    # Final comparison
    [t, P, implied_term_structure, M, I] = hull_white_main_calculation(num_paths, num_steps, end_time, a, sigma, zero_coupon_price, tolerance, seed)

    if type=="I":
        outTmp = I
    else:
        outTmp = M

    run_name = "HW-"+str(asset_id)

    multi_index_list = []
    for scenario in list(range(0,num_paths)):
        multi_index_list.append((run_name,scenario))

    multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=('Run', 'Scenario_number'))
    scenarios = pd.DataFrame(data = outTmp, columns=t, index=multi_index)

    return scenarios

