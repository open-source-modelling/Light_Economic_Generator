import os
import re
import warnings
import pytest
import numpy as np
import pandas as pd
import read_input
from read_input import validate_model_input
from black_scholes import set_up_black_scholes
from hull_white import set_up_hull_white
from vasicek import set_up_vasicek

@pytest.fixture
def parameters():
    # The example input. Every column has the type object, so that a test can put any value in it.
    return pd.read_csv(os.path.join(read_input.DATA_FOLDER, "Parameters.csv"), index_col=0).astype(object)

def test_example_input_is_valid(parameters):
    # The example input gives neither an error nor a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        validate_model_input(parameters)

@pytest.mark.parametrize("run_id, column, value, message", [
    (11, "model", "hw", "Run 11: model must be HW, BS or V, not 'hw'"),
    (11, "Type", "i", "Run 11: Type must be I (index) or D (discount factor), not 'i'"),
    (11, "NoOfPaths", 0, "Run 11: NoOfPaths must be a whole number of at least 1, not 0"),
    (11, "NoOfPaths", 10.5, "Run 11: NoOfPaths must be a whole number of at least 1, not 10.5"),
    (11, "NoOfPaths", "many", "Run 11: NoOfPaths must be a whole number of at least 1, not 'many'"),
    (22, "T", -5, "Run 22: T must be positive, not -5"),
    (22, "T", np.inf, "Run 22: T must be positive, not inf"),
    (22, "sigma", -0.2, "Run 22: sigma must be at least 0, not -0.2"),
    (22, "sigma", np.nan, "Run 22: sigma must be at least 0, not blank"),
    (11, "a", 0, "Run 11: a must be positive, not 0"),
    (11, "epsilon", 0, "Run 11: epsilon must be positive, not 0"),
    (33, "mu", np.nan, "Run 33: mu must be a number, not blank"),
    (33, "gamma", 0, "Run 33: gamma must be positive, not 0"),
    (11, "seed", -1, "Run 11: seed must be blank or a whole number of at least 0, not -1"),
    (11, "seed", 1.5, "Run 11: seed must be blank or a whole number of at least 0, not 1.5"),
    (11, "Country", "Slovenija", "Run 11: Country 'Slovenija' is not in Param_no_VA.csv. Did you mean 'Slovenia'?"),
    (11, "Country", np.nan, "Run 11: Country is blank"),
    (11, "selected_param_file", "Param_2023.csv", "Run 11: selected_param_file 'Param_2023.csv' was not found in the folder data"),
    (11, "selected_curves_file", np.nan, "Run 11: selected_curves_file is blank"),
])
def test_invalid_value(parameters, run_id, column, value, message):
    parameters.loc[run_id, column] = value
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_model_input(parameters)

def test_parameters_not_used_by_the_model_are_not_checked(parameters):
    # a is only used by Hull-White, mu and gamma only by Vasicek and epsilon not by Black-Scholes
    parameters.loc[22, ["a", "mu", "gamma", "epsilon"]] = np.nan
    parameters.loc[11, ["mu", "gamma"]] = np.nan
    parameters.loc[33, "a"] = np.nan
    validate_model_input(parameters)

def test_whole_number_written_as_decimal_is_valid(parameters):
    parameters.loc[11, ["NoOfPaths", "seed"]] = [10000.0, 1.0]
    validate_model_input(parameters)

def test_read_model_input_converts_whole_numbers(data_folder):
    # NoOfPaths and NoOfSteps written as decimals (Ex. 10000.0) are passed to the models as integers
    parameters = pd.read_csv(data_folder / "Parameters.csv", index_col=0)
    parameters[["NoOfPaths", "NoOfSteps"]] = parameters[["NoOfPaths", "NoOfSteps"]].astype(float)
    parameters.to_csv(data_folder / "Parameters.csv")
    modeling_parameters = read_input.read_model_input(11)[0]
    assert isinstance(modeling_parameters["num_paths"], int)
    assert isinstance(modeling_parameters["num_steps"], int)

def test_blank_seed_is_valid(parameters):
    parameters.loc[11, "seed"] = np.nan
    validate_model_input(parameters)

def test_seed_column_is_optional(parameters):
    validate_model_input(parameters.drop(columns="seed"))

def test_missing_column(parameters):
    with pytest.raises(ValueError, match="Parameters.csv is missing the columns: gamma"):
        validate_model_input(parameters.drop(columns="gamma"))

def test_no_runs(parameters):
    with pytest.raises(ValueError, match="Parameters.csv does not contain any runs"):
        validate_model_input(parameters.iloc[0:0])

def test_duplicate_calibration_id(parameters):
    duplicate = parameters.loc[[11]].copy()
    duplicate["seed"] = 99
    with pytest.raises(ValueError, match="Calibration_ID 11 is used by more than one run"):
        validate_model_input(pd.concat([parameters, duplicate]))

def test_different_time_grids(parameters):
    # All runs are written into one table, so a different time grid would leave blank cells
    parameters.loc[22, "NoOfSteps"] = 200
    with pytest.raises(ValueError, match="they must have the same T and NoOfSteps"):
        validate_model_input(parameters)

def test_same_seed_gives_warning(parameters):
    parameters.loc[[11, 33], "seed"] = 5
    with pytest.warns(UserWarning, match="Runs 11, 33 use the same seed 5"):
        validate_model_input(parameters)

def test_all_errors_are_reported_together(parameters):
    parameters.loc[11, "Type"] = "i"
    parameters.loc[33, "gamma"] = 0
    with pytest.raises(ValueError) as error:
        validate_model_input(parameters)
    assert "Run 11: Type" in str(error.value)
    assert "Run 33: gamma" in str(error.value)

# The output type is checked before the simulation

SET_UPS = {"BS": set_up_black_scholes, "HW": set_up_hull_white, "V": set_up_vasicek}

@pytest.mark.parametrize("model", SET_UPS)
def test_set_up_checks_output_type_before_simulation(model):
    def zero_coupon_bond_prices(t):
        raise AssertionError("The model was simulated before the output type was checked")
    modeling_parameters = {"num_paths": 10, "num_steps": 12, "end_time": 1, "a": 0.05, "mu": 0.02, "gamma": 0.3,
                           "sigma": 0.02, "tolerance": 0.01, "curve_type": "X"}
    with pytest.raises(ValueError, match=re.escape("Output type must be I (index) or D (discount factor), not 'X'")):
        SET_UPS[model](1, modeling_parameters, zero_coupon_bond_prices)
