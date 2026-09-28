import pytest
import numpy as np
from black_scholes import calculate_black_scholes_paths

# Define a simple zero-coupon bond price function for testing
@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.03 * np.asarray(t, dtype=float))  # Example yield curve: constant yield of 3%

@pytest.mark.parametrize("parameters", [
    {"num_paths": 0},
    {"num_steps": 0},
    {"end_time": 0},
    {"end_time": -5},
    {"volatility": -0.2},
])
def test_black_scholes_paths_invalid_input(zero_coupon_bond_prices, parameters):
    # Invalid parameters must raise a ValueError before any simulation
    valid = {"num_paths": 10, "num_steps": 12, "end_time": 1, "volatility": 0.2}
    valid.update(parameters)
    with pytest.raises(ValueError):
        calculate_black_scholes_paths(function_zero_coupon_price=zero_coupon_bond_prices, **valid)

def test_black_scholes_paths_zero_volatility_allowed(non_flat_curve):
    # sigma = 0 is valid and gives the deterministic index 1/P(0,t) for every path
    paths = calculate_black_scholes_paths(10, 360, 30, non_flat_curve["price"], 0.0, rng=0)
    assert np.allclose(paths["S"], 1 / non_flat_curve["price"](paths["time"]), rtol=1e-12, atol=0)
