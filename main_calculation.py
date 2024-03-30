import numpy as np
import pandas as pd
from HW_run import *



def mainCalculation(NoOfPaths, NoOfSteps, T, a, sigma, P0t, epsilon):
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
 
    paths = Paths(NoOfPaths, NoOfSteps, T, P0t, a, sigma, epsilon)
    M = paths["M"]
    t = paths["time"]
    implied_term_structure = P0t(t)
    # Compare the price of an option on a ZCB from Monte Carlo and the analytical expression
    P = np.zeros([NoOfSteps+1])
    for i in range(0, NoOfSteps+1):
        P[i] = np.mean(M[:, i])
    
    return [t, P, implied_term_structure, M]
