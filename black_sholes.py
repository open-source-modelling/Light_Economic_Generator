import numpy as np
import pandas as pd
from term_structure import calculate_instantaneous_forward_rate, calculate_zero_coupon_price

def calculate_black_sholes_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_drift: float, volatility: float, tolerance: float):
    """
    Simulates a series of stochastic interest rate paths using the Black-Sholes model.

    Args:
        num_paths (int): number of paths to simulate.
        num_steps (int): number of time steps per path.
        end_time (float): end of the modelling window (in years). 
            (Ex. a modelling window of 50 years means T=50).
        function_zero_coupon_price (function): function that calculates the price of a 
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        mean_drift (float): average drift parameter mu of the Black-Sholes model.
        volatility (float): volatility parameter sigma of the Black-Sholes model.
        tolerance (float): size of the increment used for finite 
            difference approximation.

    Returns:
        dict: A dictionary containing arrays with time steps, interest rate paths, 
            and bond prices.
            time (array): array of time steps.
            R (array): array of interest rate paths with 
              shape (num_paths, num_steps+1).
            M (array): array of bond prices with 
              shape (num_paths, num_steps+1).

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.

    Original inspiration: https://www.youtube.com/watch?v=BIZdwUDbnDo
    """       
    
    # Initial instantaneous forward rate at time t-> 0 (also spot rate at time 0).
    # r(0) = f(0,0) = - partial derivative of log(P_mkt(0, epsilon) w.r.t epsilon)
    r0 = calculate_instantaneous_forward_rate(tolerance, function_zero_coupon_price, tolerance)
        
    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])

    # Vector of time moments.
    time = np.linspace(0, end_time, num_steps+1) 
    
    W = np.zeros([num_paths, num_steps+1])
    
    # Initialize array with interest rate increments
    R = np.zeros([num_paths, num_steps+1]) 
    
    # First interest rate equals the instantaneous forward (spot) 
    # rate at time 0.
    R[:, 0] = r0 
    dt = end_time/float(num_steps) # Size of increments between two steps
    
    for iTime in range(1, num_steps+1): # For each time increment
        # Making sure the samples from the normal distribution have a mean of 0 
        # and variance 1
        if num_paths > 1:
            Z[:, iTime-1] = (Z[:, iTime-1]-np.mean(Z[:, iTime-1]))/np.std(Z[:, iTime-1])
            
        # Apply the Euler-Maruyama discretisation scheme for the Black-Sholes model
        # at each time increment.
        W[:, iTime] = W[:, iTime-1] + np.power(dt, 0.5)*Z[:, iTime-1] 
        noise_term = volatility* (W[:, iTime]-W[:, iTime-1])
        rate_term = (mean_drift-volatility**2 /2)*dt
        R[:, iTime] = R[:, iTime-1] + rate_term + noise_term
    
    # Vectorized numeric integration using the Euler integration method.
    M = np.exp(-0.5 * (R[:, :-1] + R[:, 1:]) * dt) 
    M = np.insert(M, 0, 1, axis=1).cumprod(axis=1)
    I = 1/M
    # Output is a dataframe with time moment, the interest rate path and the price
    # of a zero coupon bond issued at time 0 that matures at the selected time 
    # moment with a notional value of 1.
    paths = {"time":time, "R":R, "M":M, "I":I}
    return paths


def black_sholes_main_calculation(num_paths: int, num_steps: int, end_time: int, mean_drift: float, volatility: float, function_zero_coupon_price: callable, tolerance: float)->list:
    """
    Calculates and plots the prices of zero-coupon bonds (ZCB) calculated 
    using the Black-Sholes model`s analytical formula and the Monte Carlo simulation.
    
    Args:
        num_paths (int): number of Monte Carlo simulation paths.
        num_steps (int): number of time steps per path.
        end_time (float): end of the modelling window (in years). 
            (Ex. a modelling window of 50 years means T=50).
        function_zero_coupon_price (function): function that calculates the price of a 
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        mean_drift (float): average drift parameter mu of the Black-Sholes model.
        volatility (float): volatility parameter sigma of the Black-Sholes model.
        tolerance (float): size of the increment used for finite 
            difference approximation.
    
    Returns:
        t : time increments.
        P : average of the sumulated paths.
        implied_term_structure : term structure provided as input into the BS simulation.

        
    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.        
    """
 
    paths = calculate_black_sholes_paths(num_paths, num_steps, end_time, function_zero_coupon_price, mean_drift, volatility, tolerance)
    M = paths["M"]
    t = paths["time"]
    I = paths["I"]
    implied_term_structure = function_zero_coupon_price(t)
    # Compare the price of an option on a ZCB from Monte Carlo and the analytical expression
    P = np.zeros([num_steps+1])
    for i in range(0, num_steps+1):
        P[i] = np.mean(M[:, i])
    
    return [t, P, implied_term_structure, M, I]


def set_up_black_sholes(asset_id: int, modeling_parameters: dict, zero_coupon_price: callable)->pd.DataFrame:


    num_paths = modeling_parameters["num_paths"]  # Number of stochastic scenarios
    num_steps = modeling_parameters["num_steps"]  # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = modeling_parameters["end_time"]    # Time horizon in years (A time horizon of 50 years; T=50)
    mu =  modeling_parameters["mu"]               # Black-Sholes mean reversion parameter a
    sigma = modeling_parameters["sigma"]          # Black-Sholes volatility parameter sigma
    tolerance =  modeling_parameters["tolerance"] # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)
    type = modeling_parameters["curve_type"]

    # Final comparison
    [t, P, implied_term_structure, M, I] = black_sholes_main_calculation(num_paths, num_steps, end_time, mu, sigma, zero_coupon_price, tolerance)

    run_name = "BS-"+str(asset_id)

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
