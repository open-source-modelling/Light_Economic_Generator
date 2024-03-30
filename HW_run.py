import numpy as np
import pandas as pd
from term_structure import *

def HW_theta(a, sigma, P0t, eps):
    """
    Calculates the theta value for the Hull-White model 
    using a numeric approximation of the instantaneous forward rate 
    and the spot rate.

    Args:
        a (float): Mean reversion rate parameter a.
        sigma (float): Volatility parameter sigma.
        P0t (function handle): Function that calculates the price of a
            zero-coupon bond as a function of time.
        eps (float): Increment of time used in the numeric calculation of the 
            derivative of the instantaneous forward rate.

    Returns:
        theta (function): Function that returns the parameter theta of 
            Hull-White model at the time t implied by the calibration.
    """
    
    def theta(t):
        insta_forward_term = (f0t(t+eps, P0t, eps) 
                                         -f0t(t-eps,P0t,eps))/(2.0*eps)
                                         
        forward_term = a*f0t(t, P0t, eps)
        variance_term = sigma**2/(2.0*a)*(1.0-np.exp(-2.0*a*t))
        return insta_forward_term + forward_term + variance_term
    return theta

def Paths(NoOfPaths, NoOfSteps, T, P0t, a, sigma, epsilon):
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
    theta = HW_theta(a, sigma, P0t, epsilon)
    
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
        rate_term = (theta(time[iTime-1])-a*R[:, iTime-1])*dt
        R[:, iTime] = R[:, iTime-1] + rate_term + noise_term
    
    # Vectorized numeric integration using the Euler integration method .
    M = np.exp(-0.5 * (R[:, :-1] + R[:, 1:]) * dt) 
    M = np.insert(M, 0, 1, axis=1).cumprod(axis=1)
    
    # Output is a dataframe with time moment, the interest rate path and the price
    # of a zero coupon bond issued at time 0 that matures at the selected time 
    # moment with a notional value of 1.
    paths = {"time":time, "R":R, "M":M}
    return paths
