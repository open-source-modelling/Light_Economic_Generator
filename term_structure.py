import numpy as np
import pandas as pd
from numpy import ndarray


def smith_wilson_extrapolate_yield_curve(target_maturities: ndarray, observed_maturities: ndarray, calibration_vector: ndarray, ultimate_forward_rate: float, convergence_speed: float, tolerance: float = 0.00001) -> ndarray:
    """
    Interpolate or extrapolate rates for targeted maturities using a 
    Smith-Wilson algorithm.
       sw_extrapolate(target_maturities, observed_maturities, calibration_vector, ultimate_forward_rate, convergence_speed, tolerance) calculates the rates for 
           maturities specified in target_maturities using the calibration vector b.
    Args:
        target_maturities (ndarray): k x 1 array of targeted bond maturities.
        observed_maturities (ndarray): n x 1 array of observed bond maturities.
        calibration_vector (ndarray): n x 1 array. Calibration vector.
        ultimate_forward_rate (float): Ultimate forward rate.
        convergence_speed (float): Convergence speed parameter.
        tolerance (float): Increment to calculate the instantaneous spot rate.

    Returns:
        ndarray: k x 1 array of targeted rates for zero-coupon bonds with 
            maturity specified in target_maturities.

    For more information see 
    https://www.eiopa.europa.eu/sites/default/files/risk_free_interest_rate
        /12092019-technical_documentation.pdf
    """
    
    def smith_wilson_heart(u: ndarray, v: ndarray, convergence_speed: float) -> ndarray:
        """
        Calculate the heart of the Wilson function. sw_heart(u, v, convergence_speed) 
        calculates the matrix H (Heart of the Wilson function) for maturities 
        specified by vectors u and v. The formula is taken from the EIOPA technical 
        specifications paragraph 132.

        Args:
            u (ndarray): n_1 x 1 vector of maturities.
            v (ndarray): n_2 x 1 vector of maturities.
            convergence_speed (float): Convergence speed parameter.

        Returns:
            ndarray: n_1 x n_2 matrix representing the Heart of the Wilson function.
        """
        u_mat = np.tile(u, [v.size, 1]).transpose()
        v_mat = np.tile(v, [u.size, 1])
        return 0.5 * (convergence_speed * (u_mat + v_mat) + np.exp(-convergence_speed * (u_mat + v_mat)) 
                      - convergence_speed * np.absolute(u_mat - v_mat) - 
                      np.exp(-convergence_speed * np.absolute(u_mat - v_mat)))
    
    # Heart of the Wilson function from paragraph 132
    h = smith_wilson_heart(target_maturities, observed_maturities, convergence_speed) 
    
    # Discount pricing function for targeted maturities from paragraph 147
    p = np.exp(-np.log(1 + ultimate_forward_rate) * target_maturities) + np.diag(np.exp(-np.log(1 + ultimate_forward_rate) 
                                                     * target_maturities)) @ h @ calibration_vector 
    
    # If the first element of target_maturities is zero, replace it with time "epsilon" 
    # to avoid division by zero error.
    target_maturities[0] = tolerance if target_maturities[0] == 0 else target_maturities[0]

    return p ** (-1 / target_maturities) - 1



def calculate_zero_coupon_price(maturity, target_maturities, calibration_vector, ultimate_forward_rate: float, tolerance: float):
    """
    Calculates the price of a zero-coupon bond issued at time 0, 
    for a given maturity 'maturity', using the Smith-Wilson extrapolation technique.
    calculate_zero_coupon_price(t, m_obs, Qb, ufr, alpha)
    
    Args:
        maturity (float or ndarray): vector (or a single number) of maturities represented 
            as time fraction (Ex. for 18 months; t=1.5).
        target_maturities (ndarray): n x 1 array of observed bond maturities used for 
            calibration.
        calibration_vector (ndarray): n x 1 calibration vector of the Smith-Wilson algorithm 
           calculated on observed bonds.
        ultimate_forward_rate (float): Ultimate forward rate parameter for the Smith-Wilson algorithm.
        tolerance (float): Convergence speed parameter for the Smith-Wilson algorithm.
    
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
        price = calculate_zero_coupon_price(t, m_obs, Qb, ufr, alpha)
        print(f"Price of zero-coupon bond with maturity {t} years is: {price}")

        # For multiple maturities
        t = [1, 3, 5, 10]
        prices = calculate_zero_coupon_price(t, m_obs, Qb, ufr, alpha)
        print("Prices of zero-coupon bonds with maturities", t, "years are:")
        print(prices)

    Implemented by Gregor Fabjan from Open-Source Modelling on 29/07/2023
    """

    if isinstance(maturity, np.ndarray): # If the input is a numpy array
        y0t = smith_wilson_extrapolate_yield_curve(np.transpose(maturity), target_maturities, calibration_vector, ultimate_forward_rate, tolerance)
        price = np.exp(-y0t*np.transpose(maturity)) 
    else:# If the input is a single maturity given as a number
        y0t = smith_wilson_extrapolate_yield_curve(np.transpose([maturity]), target_maturities, calibration_vector, ultimate_forward_rate, tolerance)
        price = np.exp(-y0t*[maturity]) 
    return price

def calculate_instantaneous_forward_rate(time: float, function_zero_coupon_price: callable, tolerance: float)->float:
    """
    Calculates the instantaneous forward rate for time t given the zero-coupon
    bond price function function_zero_coupon_price, using the centered finite difference method.
    f0t(t, function_zero_coupon_price, epsilon)

    Args:
        time (float): Time at which the instantaneous forward rate is calculated.
        function_zero_coupon_price (function): Function that takes a float argument `time` and 
            returns the price of a zero-coupon bondissued at time 0 with maturity `time` and
            notional amount 1.
        tolerance (float): Step size for the centered finite difference method.

    Returns:
        float: The instantaneous forward rate at time t, calculated using the 
            centered finite difference method.
    """
    if tolerance <=0:
        raise  ValueError("Epsilon must be positive")
    
    p_plus = function_zero_coupon_price(time + tolerance)
    p_minus = function_zero_coupon_price(time - tolerance)
    return -(np.log(p_plus) - np.log(p_minus)) / (2 * tolerance)