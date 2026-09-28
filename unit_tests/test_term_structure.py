import os
import pytest
import numpy as np
import pandas as pd
import read_input
from term_structure import calculate_instantaneous_forward_rate, calculate_zero_coupon_price

# Define a simple zero-coupon bond price function for testing
@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.05 * np.asarray(t, dtype=float))  # Example yield curve: constant yield of 5%

def test_forward_rate_at_t0(zero_coupon_bond_prices):
    # Test the value of the instantaneous forward rate at t = 0
    forward_rate = calculate_instantaneous_forward_rate(0, zero_coupon_bond_prices, 0.001)
    expected_value = 0.05  # For a constant yield curve, the forward rate equals the yield
    assert pytest.approx(forward_rate, abs=1e-4) == expected_value

def test_forward_rate_at_t_positive(zero_coupon_bond_prices):
    # Test the value of the instantaneous forward rate at a positive time
    t = 1
    forward_rate = calculate_instantaneous_forward_rate(t, zero_coupon_bond_prices, 0.001)
    expected_value = -np.log(zero_coupon_bond_prices(t)) / t
    assert pytest.approx(forward_rate, abs=1e-4) == expected_value

def test_forward_rate_with_tolerance_zero(zero_coupon_bond_prices):
    # Test behavior when the tolerance is zero
    with pytest.raises(ValueError):
        calculate_instantaneous_forward_rate(1, zero_coupon_bond_prices, 0)

def test_forward_rate_on_non_flat_curve(non_flat_curve):
    # On a non-flat curve, the forward rate differs from the spot rate. The error of the
    # centered finite difference is of order epsilon^2.
    t = np.array([0, 0.5, 1, 2, 5, 10, 30])
    forward_rate = calculate_instantaneous_forward_rate(t, non_flat_curve["price"], 0.001)
    assert np.max(np.abs(forward_rate - non_flat_curve["forward"](t))) < 1e-8

# Test 1 of the validation notebooks: the term structure used by the generator reproduces
# the spot rates published by EIOPA, for every curve in the EIOPA files of Parameters.csv

def eiopa_curves():
    # Pairs of EIOPA files used in Parameters.csv, with every country of the published curves
    parameters = pd.read_csv(os.path.join(read_input.DATA_FOLDER, "Parameters.csv"), index_col=0)
    curves = []
    for param_file, curves_file in parameters[["selected_param_file", "selected_curves_file"]].drop_duplicates().values:
        countries = pd.read_csv(os.path.join(read_input.DATA_FOLDER, curves_file), index_col=0, nrows=0).columns
        curves += [pytest.param(param_file, curves_file, country, id=f"{curves_file}:{country}") for country in countries]
    return curves

@pytest.mark.parametrize("param_file, curves_file, country", eiopa_curves())
def test_term_structure_reproduces_eiopa_curve(data_folder, param_file, curves_file, country):
    # The curve is read with read_model_input, as in the generator, from a run for this country
    parameters = pd.read_csv(data_folder / "Parameters.csv", index_col=0)
    run_id = parameters.index[0]
    parameters.loc[run_id, ["selected_param_file", "selected_curves_file", "Country"]] = [param_file, curves_file, country]
    parameters.to_csv(data_folder / "Parameters.csv")
    curve_parameters = read_input.read_model_input(run_id)[1]

    published = pd.read_csv(data_folder / curves_file, index_col=0)[country]
    maturities = published.index.values.astype(float)
    price = calculate_zero_coupon_price(maturities, curve_parameters["target_maturities"], curve_parameters["calibration_vector"],
                                        curve_parameters["ultimate_forward_rate"], curve_parameters["convergence_speed"])
    spot_rates = price ** (-1 / maturities) - 1   # Annually compounded, as published by EIOPA
    difference_in_bps = np.abs(spot_rates - published.values.astype(float)) * 10000
    # Same success criteria as in the notebooks. The published rates are rounded to 0.1 bp.
    assert np.max(difference_in_bps) < 0.1
    assert np.mean(difference_in_bps) < 0.05
