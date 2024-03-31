import numpy as np
import pandas as pd
from term_structure import *

def Paths_BS(NoOfPaths, NoOfSteps, T, P0t, mu, sigma, epsilon):
    """
    Simulates a series of stochastic interest rate paths using the Hull-White model.

    Args:
        NoOfPaths (int): number of paths to simulate.
        NoOfSteps (int): number of time steps per path.
        T (float): end of the modelling window (in years). 
            (Ex. a modelling window of 50 years means T=50).
        P0t (function): function that calculates the price of a 
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        a (float): mean reversion speed parameter a of 
            the Hull-White model.
        sigma (float): volatility parameter sigma of the Hull-White model.
        epsilon (float): size of the increment used for finite 
            difference approximation.

    Returns:
        dict: A dictionary containing arrays with time steps, interest rate paths, 
            and bond prices.
            time (array): array of time steps.
            R (array): array of interest rate paths with 
              shape (NoOfPaths, NoOfSteps+1).
            M (array): array of bond prices with 
              shape (NoOfPaths, NoOfSteps+1).

    Implemented by Gregor Fabjan from Open-Source Modelling on 29/07/2023.

    Original inspiration: https://www.youtube.com/watch?v=BIZdwUDbnDo
    """       
    
    # Initial instantaneous forward rate at time t-> 0 (also spot rate at time 0).
    # r(0) = f(0,0) = - partial derivative of log(P_mkt(0, epsilon) w.r.t epsilon)
    r0 = f0t(epsilon, P0t, epsilon)
    
    # Calculation of theta = 1/a * partial derivative of f(0,t) w.r.t. t 
    # + f(0,t) + sigma^2/(2 a^2)* (1-exp(-2*a*t)).
    #theta = HW_theta(a, sigma, P0t, epsilon)
    
    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])

    # Initialize arrays
    
    # Vector of time moments.
    time = np.linspace(0, T, NoOfSteps+1) 
    
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    
    # Initialize array with interest rate increments
    R = np.zeros([NoOfPaths, NoOfSteps+1]) 
    
    # First interest rate equals the instantaneous forward (spot) 
    # rate at time 0.
    R[:, 0] = r0 
    dt = T/float(NoOfSteps) # Size of increments between two steps
    
    for iTime in range(1, NoOfSteps+1): # For each time increment
        # Making sure the samples from the normal distribution have a mean of 0 
        # and variance 1
        if NoOfPaths > 1:
            Z[:, iTime-1] = (Z[:, iTime-1]-np.mean(Z[:, iTime-1]))/np.std(Z[:, iTime-1])
            
        # Apply the Euler-Maruyama discretisation scheme for the Hull-White model
        # at each time increment.
        W[:, iTime] = W[:, iTime-1] + np.power(dt, 0.5)*Z[:, iTime-1] 
        noise_term = sigma* (W[:, iTime]-W[:, iTime-1])
        rate_term = (mu-sigma**2 /2)*dt
        R[:, iTime] = R[:, iTime-1] + rate_term + noise_term
    
    # Vectorized numeric integration using the Euler integration method .
    M = np.exp(-0.5 * (R[:, :-1] + R[:, 1:]) * dt) 
    M = np.insert(M, 0, 1, axis=1).cumprod(axis=1)
    
    # Output is a dataframe with time moment, the interest rate path and the price
    # of a zero coupon bond issued at time 0 that matures at the selected time 
    # moment with a notional value of 1.
    paths = {"time":time, "R":R, "M":M}
    return paths


def mainCalculation_BS(NoOfPaths, NoOfSteps, T, mu, sigma, P0t, epsilon):
    """
    Calculates and plots the prices of zero-coupon bonds (ZCB) calculated 
    using the Hull-White model`s analytical formula and the Monte Carlo simulation.
    
    Args:
        NoOfPaths (int): number of Monte Carlo simulation paths.
        NoOfSteps (int): number of time steps per path.
        T (float): length in years of the modelling window (Ex. 50 years means t=50).
        a (float): mean reversion rate parameter a of the Hull-White model.
        sigma (float): volatility parameter sigma of the Hull-White model.
        P0t (function): function that calculates the price of a zero coupon bond issued. 
           at time 0 that matures at time t, with a notional amount 1 and discounted using
           the assumed term structure.
        epsilon (float): the size of the increment  used for finite difference approximation.
    
    Returns:
        t XXX: time increments.
        P XXX: average of the sumulated paths.
        implied_term_structure XXX: term structure provided as input into the HW simulation.
    """
 
    paths = Paths_BS(NoOfPaths, NoOfSteps, T, P0t, mu, sigma, epsilon)
    M = paths["M"]
    t = paths["time"]
    implied_term_structure = P0t(t)
    # Compare the price of an option on a ZCB from Monte Carlo and the analytical expression
    P = np.zeros([NoOfSteps+1])
    for i in range(0, NoOfSteps+1):
        P[i] = np.mean(M[:, i])
    
    return [t, P, implied_term_structure, M]


def Run_Black_Sholes(asset_id):
    P0t = lambda t: P0t_f(t, m_obs, Qb, ufr, alpha)

    param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

    selected_param_file = param_raw["selected_param_file"][asset_id]
    selected_curves_file = param_raw["selected_curves_file"][asset_id]
    country = param_raw["Country"][asset_id]

    NoOfPaths = param_raw["NoOfPaths"][asset_id] # Number of stochastic scenarios
    NoOfSteps = param_raw["NoOfSteps"][asset_id] # Number of equidistand discrete modelling points (50*12 = 600)
    T = param_raw["T"][asset_id]                 # Time horizon in years (A time horizon of 50 years; T=50)
    mu =  param_raw["mu"][asset_id]                # Hull-White mean reversion parameter a
    sigma = param_raw["sigma"][asset_id]         # Hull-White volatility parameter sigma
    epsilon =  param_raw["epsilon"][asset_id]     # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)

    param_raw = pd.read_csv(selected_param_file, sep=',', index_col=0)

    maturities_country_raw = param_raw.loc[:,country+"_Maturities"].iloc[6:]
    param_country_raw = param_raw.loc[:,country + "_Values"].iloc[6:]
    extra_param = param_raw.loc[:,country + "_Values"].iloc[:6]

    relevant_positions = pd.notna(maturities_country_raw.values)
    maturities_country = maturities_country_raw.iloc[relevant_positions]
    Qb = param_country_raw.iloc[relevant_positions]
    curve_raw = pd.read_csv(selected_curves_file, sep=',',index_col=0)
    curve_country = curve_raw.loc[:,country]

    # Maturity of observations:
    m_obs = np.transpose(np.array(maturities_country.values))

    # Ultimate froward rate ufr represents the rate to which the rate curve will 
    # converge as time increases:
    ufr = extra_param.iloc[3]/100

    # Convergence speed parameter alpha controls the speed at which the curve 
    # converges towards the ufr from the last liquid point:
    alpha = extra_param.iloc[4]

    # For which maturities do we want the SW algorithm to calculate the rates. 
    # In this case, for every year up to 150:
    m_target = np.transpose(np.arange(1,151)) 

    # Qb calibration vector published by EIOPA for the curve calibration:
    Qb = np.transpose(np.array(Qb.values))

    # Final comparison
    [t, P, implied_term_structure, M] = mainCalculation_BS(NoOfPaths, NoOfSteps, T, mu, sigma, P0t, epsilon)

    run_name = "BS-"+str(asset_id)

    multi_index_list = []
    for scenario in list(range(0,NoOfPaths)):
        multi_index_list.append((run_name,scenario))

    multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=('Run', 'Scenario_number'))
    out = pd.DataFrame(data = M,columns=t,index=multi_index)

    return out
