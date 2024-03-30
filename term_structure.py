
import numpy as np
import pandas as pd

def sw_extrapolate(m_target, m_obs, Qb, ufr, alpha, epsilon = 0.00001):
    """
    Interpolate or extrapolate rates for targeted maturities using a 
    Smith-Wilson algorithm.
       sw_extrapolate(m_target, m_obs, Qb, ufr, alpha, epsilon) calculates the rates for 
           maturities specified in M_Target using the calibration vector b.
    Args:
        m_target (ndarray): k x 1 array of targeted bond maturities.
        m_obs (ndarray): n x 1 array of observed bond maturities.
        Qb (ndarray): n x 1 array. Calibration vector.
        ufr (float): Ultimate forward rate.
        alpha (float): Convergence speed parameter.
        epsilon (float): Increment to calculate the instantaneous spot rate.

    Returns:
        ndarray: k x 1 array of targeted rates for zero-coupon bonds with 
            maturity specified in m_target.

    For more information see 
    https://www.eiopa.europa.eu/sites/default/files/risk_free_interest_rate
        /12092019-technical_documentation.pdf
    """
    
    def sw_heart(u, v, alpha):
        """
        Calculate the heart of the Wilson function. sw_heart(u, v, alpha) 
        calculates the matrix H (Heart of the Wilson function) for maturities 
        specified by vectors u and v. The formula is taken from the EIOPA technical 
        specifications paragraph 132.

        Args:
            u (ndarray): n_1 x 1 vector of maturities.
            v (ndarray): n_2 x 1 vector of maturities.
            alpha (float): Convergence speed parameter.

        Returns:
            ndarray: n_1 x n_2 matrix representing the Heart of the Wilson function.
        """
        u_mat = np.tile(u, [v.size, 1]).transpose()
        v_mat = np.tile(v, [u.size, 1])
        return 0.5 * (alpha * (u_mat + v_mat) + np.exp(-alpha * (u_mat + v_mat)) 
                      - alpha * np.absolute(u_mat - v_mat) - 
                      np.exp(-alpha * np.absolute(u_mat - v_mat)))
    
    # Heart of the Wilson function from paragraph 132
    h = sw_heart(m_target, m_obs, alpha) 
    
    # Discount pricing function for targeted maturities from paragraph 147
    p = np.exp(-np.log(1 + ufr) * m_target) + np.diag(np.exp(-np.log(1 + ufr) 
                                                     * m_target)) @ h @ Qb 
    
    # If the first element of m_target is zero, replace it with time "epsilon" 
    # to avoid division by zero error.
    m_target[0] = epsilon if m_target[0] == 0 else m_target[0]

    return p ** (-1 / m_target) - 1



def P0t_f(t, m_obs, Qb, ufr, alpha):
    """
    Calculates the price of a zero-coupon bond issued at time 0, 
    for a given maturity 't', using the Smith-Wilson extrapolation technique.
    P0t_f(t, m_obs, Qb, ufr, alpha)
    
    Args:
        t (float or ndarray): vector (or a single number) of maturities represented 
            as time fraction (Ex. for 18 months; t=1.5).
        m_obs (ndarray): n x 1 array of observed bond maturities used for 
            calibration.
        Qb (ndarray): n x 1 calibration vector of the Smith-Wilson algorithm 
           calculated on observed bonds.
        ufr (float): Ultimate forward rate parameter for the Smith-Wilson algorithm.
        alpha (float): Convergence speed parameter for the Smith-Wilson algorithm.
    
    Returns:
        ndarray: n x 1 the price of zero-coupon bonds issued at time 0 with a notional amount of 1
            and maturity t.   
        
    Example of use
        m_obs = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30])
        Qb = np.array([0.02474805, 0.02763133, 0.02926931, 0.0302894, 0.03061605,
           0.03068016, 0.03038397, 0.02999401, 0.02926168])
        ufr = 0.042
        alpha = 0.05        

        # For a single maturity
        t = 5
        price = P0t_f(t, m_obs, Qb, ufr, alpha)
        print(f"Price of zero-coupon bond with maturity {t} years is: {price}")

        # For multiple maturities
        t = [1, 3, 5, 10]
        prices = P0t_f(t, m_obs, Qb, ufr, alpha)
        print("Prices of zero-coupon bonds with maturities", t, "years are:")
        print(prices)

    Implemented by Gregor Fabjan from Open-Source Modelling on 29/07/2023
    """

    if isinstance(t, np.ndarray): # If the input is a numpy array
        y0t = sw_extrapolate(np.transpose(t), m_obs, Qb, ufr, alpha)
        out = np.exp(-y0t*np.transpose(t)) 
    else:# If the input is a single maturity given as a number
        y0t = sw_extrapolate(np.transpose([t]), m_obs, Qb, ufr, alpha)
        out = np.exp(-y0t*[t]) 
    return out

def f0t(t, P0t, epsilon):
    """
    Calculates the instantaneous forward rate for time t given the zero-coupon
    bond price function P0t, using the centered finite difference method.
    f0t(t, P0t, epsilon)

    Args:
        t (float): Time at which the instantaneous forward rate is calculated.
        P0t (function): Function that takes a float argument `t` and 
            returns the price of a zero-coupon bondissued at time 0 with maturity `t` and
            notional amount 1.
        epsilon (float): Step size for the centered finite difference method.

    Returns:
        float: The instantaneous forward rate at time t, calculated using the 
            centered finite difference method.
    """
    
    p_plus = P0t(t + epsilon)
    p_minus = P0t(t - epsilon)
    return -(np.log(p_plus) - np.log(p_minus)) / (2 * epsilon)