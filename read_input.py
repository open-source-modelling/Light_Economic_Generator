import pandas as pd
import numpy as np

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


