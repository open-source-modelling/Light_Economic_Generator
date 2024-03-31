import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from term_structure import *
from HW_run import *
from IPython.display import display

asset_id = 11

def Run_Hull_White(asset_id):
    # Zero coupon bond prices calculated using the assumed term structure
    P0t = lambda t: P0t_f(t, m_obs, Qb, ufr, alpha)

    param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

    selected_param_file = param_raw["selected_param_file"][asset_id]
    selected_curves_file = param_raw["selected_curves_file"][asset_id]
    country = param_raw["Country"][asset_id]

    NoOfPaths = param_raw["NoOfPaths"][asset_id] # Number of stochastic scenarios
    NoOfSteps = param_raw["NoOfSteps"][asset_id] # Number of equidistand discrete modelling points (50*12 = 600)
    T = param_raw["T"][asset_id]                 # Time horizon in years (A time horizon of 50 years; T=50)
    a =  param_raw["a"][asset_id]                # Hull-White mean reversion parameter a
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
    [t, P, implied_term_structure, M] = mainCalculation(NoOfPaths, NoOfSteps, T, a, sigma, P0t, epsilon)

    run_name = "HW-"+str(asset_id)

    multi_index_list = []
    for scenario in list(range(0,NoOfPaths)):
        multi_index_list.append((run_name,scenario))

    multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=('Run', 'Scenario_number'))
    out = pd.DataFrame(data = M,columns=t,index=multi_index)

    return out


print(Run_Hull_White(11))
