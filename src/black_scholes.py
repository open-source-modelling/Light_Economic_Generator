import numpy as np
import pandas as pd

def calculate_black_scholes_paths(num_paths: int, num_steps: int, end_time: int, function_zero_coupon_price: callable, volatility: float) -> dict:
    """
    Simulates a series of stochastic equity index paths using the Black-Scholes model
    under the risk-neutral measure. The drift of the index is the deterministic
    short rate implied by the term structure:

        dS(t) = r(t) S(t) dt + sigma S(t) dW(t),   S(0) = 1,   r(t) = f(0,t)

    The paths are generated with the exact solution on the time grid:

        S(t_i) = S(t_i-1) * P(0,t_i-1)/P(0,t_i) * exp(-sigma^2/2 dt + sigma dW_i)

    so that the discounted index P(0,t) * S(t) is a martingale with expectation 1.

    Args:
        num_paths (int): number of paths to simulate.
        num_steps (int): number of time steps per path.
        end_time (float): end of the modelling window (in years).
            (Ex. a modelling window of 50 years means T=50).
        function_zero_coupon_price (function): function that calculates the price of a
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.
        volatility (float): volatility parameter sigma of the Black-Scholes model.

    Returns:
        dict: A dictionary containing arrays with time steps, index paths,
            and discount factors.
            time (array): array of time steps.
            S (array): array of equity index paths with
              shape (num_paths, num_steps+1).
            M (array): array of discount factors P(0,t) with
              shape (num_paths, num_steps+1). Identical for every path.
            I (array): equity index, equal to S.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.
    """

    # Generate the single source of random noise.
    Z = np.random.normal(0.0, 1.0, [num_paths, num_steps])

    # Making sure the samples from the normal distribution have a mean of 0
    # and variance 1 at each time increment.
    if num_paths > 1:
        Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0)

    # Vector of time moments.
    time = np.linspace(0, end_time, num_steps+1)
    dt = end_time/float(num_steps) # Size of increments between two steps

    # Deterministic discount factors P(0,t) from the term structure.
    discount_factor = function_zero_coupon_price(time)

    # Risk-free growth over each time increment, P(0,t_i-1)/P(0,t_i).
    risk_free_growth = discount_factor[:-1] / discount_factor[1:]

    # Random shock over each time increment, with expectation 1.
    shock = np.exp(-0.5 * volatility**2 * dt + volatility * np.power(dt, 0.5) * Z)

    # Equity index starting at 1.
    S = np.ones([num_paths, num_steps+1])
    S[:, 1:] = np.cumprod(risk_free_growth * shock, axis=1)

    M = np.tile(discount_factor, (num_paths, 1))
    I = S
    paths = {"time":time, "S":S, "M":M, "I":I}
    return paths


def black_scholes_main_calculation(num_paths: int, num_steps: int, end_time: int, volatility: float, function_zero_coupon_price: callable) -> list:
    """
    Simulates the Black-Scholes equity index and calculates the average
    discounted index, which should be equal to 1 at every time step.

    Args:
        num_paths (int): number of Monte Carlo simulation paths.
        num_steps (int): number of time steps per path.
        end_time (float): end of the modelling window (in years).
            (Ex. a modelling window of 50 years means T=50).
        volatility (float): volatility parameter sigma of the Black-Scholes model.
        function_zero_coupon_price (function): function that calculates the price of a
            zero coupon bond issued at time 0 that matures at time t, with a
            notional amount 1 and discounted using the assumed term structure.

    Returns:
        t : time increments.
        P : average of the discounted index paths (martingale test, should be 1).
        implied_term_structure : term structure provided as input into the BS simulation.
        M : discount factors.
        I : equity index paths.

    Implemented by Gregor Fabjan from Open-Source Modelling on 13/04/2024.
    """

    paths = calculate_black_scholes_paths(num_paths, num_steps, end_time, function_zero_coupon_price, volatility)
    M = paths["M"]
    t = paths["time"]
    I = paths["I"]
    implied_term_structure = function_zero_coupon_price(t)
    P = np.mean(M * I, axis=0)

    return [t, P, implied_term_structure, M, I]


def set_up_black_scholes(asset_id: int, modeling_parameters: dict, zero_coupon_price: callable)->pd.DataFrame:


    num_paths = modeling_parameters["num_paths"]  # Number of stochastic scenarios
    num_steps = modeling_parameters["num_steps"]  # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = modeling_parameters["end_time"]    # Time horizon in years (A time horizon of 50 years; T=50)
    sigma = modeling_parameters["sigma"]          # Black-Scholes volatility parameter sigma
    type = modeling_parameters["curve_type"]

    # Final comparison
    [t, P, implied_term_structure, M, I] = black_scholes_main_calculation(num_paths, num_steps, end_time, sigma, zero_coupon_price)

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
