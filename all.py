import pandas as pd
import numpy as np
from numpy import ndarray
def main():
    param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

    combined_run = []

    for run_id in param_raw.index:
        [modeling_parameters, curve_parameters] = read_model_input(run_id)
        
        zero_coupon_price = lambda t: calculate_zero_coupon_price(t, curve_parameters["target_maturities"], curve_parameters["calibration_vector"], curve_parameters["ultimate_forward_rate"], curve_parameters["convergence_speed"] )
        if modeling_parameters["run_type"] == "HW":
            run = set_up_hull_white(run_id, modeling_parameters, zero_coupon_price)        
        elif modeling_parameters["run_type"] == "BS":
            run = set_up_black_sholes(run_id, modeling_parameters, zero_coupon_price)
        elif modeling_parameters["run_type"] == "V":
            run = set_up_vasicek(run_id, modeling_parameters, zero_coupon_price)
        else:
            raise ValueError("Model type not available")

        if isinstance(combined_run,pd.DataFrame):
            combined_run = pd.concat([combined_run,run])
        else:
            combined_run = run
    combined_run.to_csv("Output/run.csv")
def calculate_black_sholes_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_drift: float, volatility: float, tolerance: float):
     # Initial instantaneous forward rate at time t-> 0 (also spot rate at time 0).
    # r(0) = f(0,0) = - partial derivative of log(P_mkt(0, epsilon) w.r.t epsilon)
    r0 = calculate_instantaneous_forward_rate(tolerance, function_zero_coupon_price, tolerance)
        
    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])

    # Initialize arrays
    
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

def calculate_hull_white_theta(mean_reversion_rate: float, volatility: float, function_zero_coupon_price: callable, tolerance: float) -> callable:
    def theta(t:float)->float:
        insta_forward_term = (calculate_instantaneous_forward_rate(t+tolerance, function_zero_coupon_price, tolerance) 
                                         -calculate_instantaneous_forward_rate(t-tolerance,function_zero_coupon_price,tolerance))/(2.0*tolerance)
                                         
        forward_term = mean_reversion_rate*calculate_instantaneous_forward_rate(t, function_zero_coupon_price, tolerance)
        variance_term = volatility**2/(2.0*mean_reversion_rate)*(1.0-np.exp(-2.0*mean_reversion_rate*t))
        return insta_forward_term + forward_term + variance_term
    return theta

def calculate_hull_white_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_reversion_rate: float, volatility: float, tolerance: float)->dict:
    # Initial instantaneous forward rate at time t-> 0 (also spot rate at time 0).
    # r(0) = f(0,0) = - partial derivative of log(P_mkt(0, epsilon) w.r.t epsilon)
    r0 = calculate_instantaneous_forward_rate(tolerance, function_zero_coupon_price, tolerance)
    # Calculation of theta = 1/a * partial derivative of f(0,t) w.r.t. t 
    # + f(0,t) + sigma^2/(2 a^2)* (1-exp(-2*a*t)).
    theta = calculate_hull_white_theta(mean_reversion_rate, volatility, function_zero_coupon_price, tolerance)
    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])
    # Initialize arrays
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
            
        # Apply the Euler-Maruyama discretisation scheme for the Hull-White model
        # at each time increment.
        W[:, iTime] = W[:, iTime-1] + np.power(dt, 0.5)*Z[:, iTime-1] 
        noise_term = volatility* (W[:, iTime]-W[:, iTime-1])
        rate_term = (theta(time[iTime-1])-mean_reversion_rate*R[:, iTime-1])*dt
        R[:, iTime] = R[:, iTime-1] + rate_term + noise_term
    # Vectorized numeric integration using the Euler integration method .
    M = np.exp(-0.5 * (R[:, :-1] + R[:, 1:]) * dt) 
    M = np.insert(M, 0, 1, axis=1).cumprod(axis=1)
    I = 1/M
    # Output is a dataframe with time moment, the interest rate path and the price
    # of a zero coupon bond issued at time 0 that matures at the selected time 
    # moment with a notional value of 1.
    paths = {"time":time, "R":R, "M":M, "I":I}
    return paths
def hull_white_main_calculation(num_paths: int, num_steps: int, end_time: int, mean_reversion_rate: float, volatility:float, function_zero_coupon_price: callable, tolerance: float):
    paths = calculate_hull_white_paths(num_paths, num_steps, end_time, function_zero_coupon_price, mean_reversion_rate, volatility, tolerance)
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

    # Final comparison
    [t, P, implied_term_structure, M, I] = hull_white_main_calculation(num_paths, num_steps, end_time, a, sigma, zero_coupon_price, tolerance)
    if type=="I":
        outTmp = I
    elif type=="D":
        outTmp = M
    else:
        raise ValueError

    run_name = "HW-"+str(asset_id)

    multi_index_list = []
    for scenario in list(range(0,num_paths)):
        multi_index_list.append((run_name,scenario))

    multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=('Run', 'Scenario_number'))
    scenarios = pd.DataFrame(data = outTmp, columns=t, index=multi_index)

    return scenarios

def calculate_vasicek_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, mean_drift: float, sigma: float, gamma: float, tolerance: float)->dict:
    r0 = calculate_instantaneous_forward_rate(tolerance, function_zero_coupon_price, tolerance)
        
    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])

    # Initialize arrays
    
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
            
        # Apply the Euler-Maruyama discretisation scheme for the Vasicek model
        # at each time increment.
        sd_term = np.power(sigma**2 /(2*gamma)*(1-np.exp(-2*gamma*dt)),0.5)
        
        W[:, iTime] = W[:, iTime-1] + sd_term*Z[:, iTime-1] 
        noise_term = np.exp(-gamma * dt) 
        rate_term = mean_drift*(1-np.exp(-gamma*dt))
    
        R[:, iTime] = R[:, iTime-1]*noise_term + rate_term + (W[:, iTime]-W[:, iTime-1])
    # Vectorized numeric integration using the Euler integration method .
    M = np.exp(-0.5 * (R[:, :-1] + R[:, 1:]) * dt) 
    M = np.insert(M, 0, 1, axis=1).cumprod(axis=1)
    I = 1/M
    paths = {"time":time, "R":R, "M":M, "I":I}
    return paths
def vasicek_main_calculation(num_paths: int, num_steps: int, end_time: int, mean_drift: float, sigma: float, gamma: float, function_zero_coupon_price: callable, tolerance: float)-> list:
    """
    Calculates and plots the prices of zero-coupon bonds (ZCB) calculated 
    using the Vasicek model`s analytical formula and the Monte Carlo simulation.
    
    Args:
        num_paths (int): number of Monte Carlo simulation paths.
        NoOfSteps (int): number of time steps per path.
        end_time (int): length in years of the modelling window (Ex. 50 years means t=50).
        mean_drift (float): average drift parameter mu of the Vasicek model.
        sigma (float): volatility parameter sigma of the Vasicek model.
        gamma (float): parameter gamma of the Vasicek model
        function_zero_coupon_price (function): function that calculates the price of a zero coupon bond issued. 
           at time 0 that matures at time t, with a notional amount 1 and discounted using
           the assumed term structure.
        tolerance (float): the size of the increment  used for finite difference approximation.
    
    Returns:
        t : time increments.
        P : average of the sumulated paths.
        implied_term_structure : term structure provided as input into the V simulation.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.        
    """
    paths = calculate_vasicek_paths(num_paths, num_steps, end_time, function_zero_coupon_price, mean_drift, sigma, gamma, tolerance)
    M = paths["M"]
    t = paths["time"]
    I = paths["I"]
    implied_term_structure = function_zero_coupon_price(t)
    # Compare the price of an option on a ZCB from Monte Carlo and the analytical expression
    P = np.zeros([num_steps+1])
    for i in range(0, num_steps+1):
        P[i] = np.mean(M[:, i])
    
    return [t, P, implied_term_structure, M, I]
def set_up_vasicek(asset_id: int, modeling_parameters: dict, zero_coupon_price: callable) -> pd.DataFrame:


    num_paths = modeling_parameters["num_paths"]  # Number of stochastic scenarios
    num_steps = modeling_parameters["num_steps"]  # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = modeling_parameters["end_time"]    # Time horizon in years (A time horizon of 50 years; T=50)
    mu =  modeling_parameters["mu"]               # Vasicek long term mean parameter mu
    gamma = modeling_parameters["gamma"]          # Vasicek parameter gamma
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
def read_model_input(asset_id: int)->list:

    param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

    selected_param_file = param_raw["selected_param_file"][asset_id]
    selected_curves_file = param_raw["selected_curves_file"][asset_id]
    country = param_raw["Country"][asset_id]

    run_type = param_raw["model"][asset_id]
    num_paths = param_raw["NoOfPaths"][asset_id] # Number of stochastic scenarios
    num_steps = param_raw["NoOfSteps"][asset_id] # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = param_raw["T"][asset_id]                 # Time horizon in years (A time horizon of 50 years; T=50)
    a =  param_raw["a"][asset_id]                # Hull-White mean reversion parameter a
    mu =  param_raw["mu"][asset_id]                # Hull-White mean reversion parameter a
    sigma = param_raw["sigma"][asset_id]         # Hull-White volatility parameter sigma
    gamma = param_raw["gamma"][asset_id]         # Hull-White volatility parameter sigma
    tolerance =  param_raw["epsilon"][asset_id]     # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)
    curve_type = param_raw["Type"][asset_id]

    param_raw = pd.read_csv(selected_param_file, sep=',', index_col=0)

    maturities_country_raw = param_raw.loc[:,country+"_Maturities"].iloc[6:]
    param_country_raw = param_raw.loc[:,country + "_Values"].iloc[6:]
    extra_param = param_raw.loc[:,country + "_Values"].iloc[:6]

    relevant_positions = pd.notna(maturities_country_raw.values)
    maturities_country = maturities_country_raw.iloc[relevant_positions]
    calibration_vector = param_country_raw.iloc[relevant_positions]
    curve_raw = pd.read_csv(selected_curves_file, sep=',',index_col=0)
    curve_country = curve_raw.loc[:,country]

    # Curve related parameters
    target_maturities = np.transpose(np.array(maturities_country.values))
    ultimate_forward_rate = extra_param.iloc[3]/100
    convergence_speed = extra_param.iloc[4]
    calibration_vector = np.transpose(np.array(calibration_vector.values))
    curve = {"target_maturities":target_maturities, "ultimate_forward_rate":ultimate_forward_rate, "calibration_vector":calibration_vector, "convergence_speed":convergence_speed}
    modeling_run ={"num_paths":num_paths, "num_steps":num_steps,"end_time":end_time, "mu":mu, "a":a, "sigma":sigma, "gamma": gamma, "tolerance":tolerance, "curve_type":curve_type, "run_type":run_type}
    return [modeling_run, curve]


