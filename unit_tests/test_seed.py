import os
import pytest
import numpy as np
import pandas as pd
import read_input
from black_scholes import calculate_black_scholes_paths, set_up_black_scholes
from hull_white import calculate_hull_white_paths, set_up_hull_white
from vasicek import calculate_vasicek_paths, set_up_vasicek

# Define a simple zero-coupon bond price function for testing
def zero_coupon_bond_prices(t):
    return np.exp(-0.03 * np.asarray(t, dtype=float))  # Example yield curve: constant yield of 3%

# Paths of each model (50 paths, 12 steps, T = 1) for a seed or random number generator rng
SIMULATIONS = {
    "BS": lambda rng: calculate_black_scholes_paths(50, 12, 1, zero_coupon_bond_prices, 0.2, rng=rng),
    "HW": lambda rng: calculate_hull_white_paths(50, 12, 1, zero_coupon_bond_prices, 0.05, 0.008, 0.01, rng=rng),
    "V": lambda rng: calculate_vasicek_paths(50, 12, 1, zero_coupon_bond_prices, 0.02, 0.02, 0.3, 0.01, rng=rng),
}

SET_UPS = {"BS": set_up_black_scholes, "HW": set_up_hull_white, "V": set_up_vasicek}

def assert_same_paths(first, second):
    for key in first:
        assert np.array_equal(first[key], second[key]), key

@pytest.mark.parametrize("model", SIMULATIONS)
def test_same_seed_gives_same_paths(model):
    assert_same_paths(SIMULATIONS[model](1), SIMULATIONS[model](1))

@pytest.mark.parametrize("model", SIMULATIONS)
def test_different_seeds_give_different_paths(model):
    assert not np.array_equal(SIMULATIONS[model](1)["I"], SIMULATIONS[model](2)["I"])

@pytest.mark.parametrize("model", SIMULATIONS)
def test_generator_instead_of_seed(model):
    # A numpy random number generator can be passed instead of a seed
    assert_same_paths(SIMULATIONS[model](7), SIMULATIONS[model](np.random.default_rng(7)))

@pytest.mark.parametrize("model", SIMULATIONS)
def test_seed_is_independent_of_global_random_state(model):
    # With a seed, the paths do not depend on np.random.seed, and the global random state is not used
    np.random.seed(0)
    first = SIMULATIONS[model](1)
    first_global_number = np.random.random()
    np.random.seed(123)
    assert_same_paths(first, SIMULATIONS[model](1))
    np.random.seed(0)
    assert np.random.random() == first_global_number

@pytest.mark.parametrize("model", SIMULATIONS)
def test_without_seed_global_random_state_is_used(model):
    # Without a seed, the paths follow np.random.seed, as in the validation notebooks
    np.random.seed(0)
    first = SIMULATIONS[model](None)
    np.random.seed(0)
    assert_same_paths(first, SIMULATIONS[model](None))

@pytest.mark.parametrize("model", SET_UPS)
def test_set_up_uses_seed(model):
    # Two runs with the same seed give the same scenarios
    modeling_parameters = {"num_paths": 50, "num_steps": 12, "end_time": 1, "a": 0.05, "mu": 0.02, "gamma": 0.3,
                           "sigma": 0.02, "tolerance": 0.01, "curve_type": "I", "seed": 1}
    first = SET_UPS[model](1, modeling_parameters, zero_coupon_bond_prices)
    second = SET_UPS[model](1, modeling_parameters, zero_coupon_bond_prices)
    assert first.equals(second)

# Reading the seed from Parameters.csv (the fixture data_folder is defined in conftest.py)

def test_read_model_input_reads_seed():
    parameters = pd.read_csv(os.path.join(read_input.DATA_FOLDER, "Parameters.csv"), index_col=0)
    for run_id in parameters.index:
        seed = read_input.read_model_input(run_id)[0]["seed"]
        assert isinstance(seed, int)
        assert seed == parameters.loc[run_id, "seed"]

def test_read_model_input_blank_seed(data_folder):
    # A blank seed gives None; the other rows are still read as integers
    parameters = pd.read_csv(data_folder / "Parameters.csv", index_col=0)
    parameters.loc[11, "seed"] = np.nan
    parameters.to_csv(data_folder / "Parameters.csv")
    assert read_input.read_model_input(11)[0]["seed"] is None
    assert isinstance(read_input.read_model_input(22)[0]["seed"], int)

def test_read_model_input_without_seed_column(data_folder):
    # Input files without the column "seed" can still be used
    parameters = pd.read_csv(data_folder / "Parameters.csv", index_col=0)
    parameters.drop(columns="seed").to_csv(data_folder / "Parameters.csv")
    assert read_input.read_model_input(11)[0]["seed"] is None
