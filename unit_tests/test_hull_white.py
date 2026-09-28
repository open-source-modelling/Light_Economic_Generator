import pytest
import numpy as np
from hull_white import calculate_hull_white_theta, calculate_hull_white_paths

# Define a simple zero-coupon bond price function for testing
@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.05 * np.asarray(t, dtype=float))  # Example yield curve: constant yield of 5%

def test_hull_white_theta_at_t0(zero_coupon_bond_prices):
    # Test the value of theta at t = 0
    theta = calculate_hull_white_theta(0.1, 0.01, zero_coupon_bond_prices, 0.001)
    # Flat curve: the derivative of the forward rate is 0 and the variance term is 0 at t = 0,
    # so theta(0) = a * f(0,0) = 0.1 * 0.05
    expected_value = 0.005
    assert pytest.approx(theta(0), abs=1e-4) == expected_value

def test_hull_white_theta_at_t_positive(zero_coupon_bond_prices):
    # Test the value of theta at a positive time
    theta = calculate_hull_white_theta(0.1, 0.01, zero_coupon_bond_prices, 0.001)
    expected_value = 0.1 * -np.log(zero_coupon_bond_prices(1)) / 1 + \
                     0.01**2 / (2 * 0.1) * (1 - np.exp(-2 * 0.1 * 1))
    assert pytest.approx(theta(1), abs=1e-4) == expected_value

def test_hull_white_paths_martingale(zero_coupon_bond_prices):
    # The average simulated discount factor should reproduce the input term structure
    np.random.seed(0)
    paths = calculate_hull_white_paths(2000, 240, 20, zero_coupon_bond_prices, 0.1, 0.01, 0.01)
    simulated = np.mean(paths["M"], axis=0)
    expected = zero_coupon_bond_prices(paths["time"])
    assert np.max(np.abs(simulated / expected - 1)) < 0.01

def test_hull_white_theta_on_non_flat_curve(non_flat_curve):
    # theta(t) = df(0,t)/dt + a f(0,t) + sigma^2/(2a) (1 - exp(-2at)). On a non-flat curve, every
    # term matters: the derivative of the forward rate is not 0 and at t = 1 the variance term
    # (9e-5) is far above the tolerance.
    a, sigma = 0.1, 0.01
    theta = calculate_hull_white_theta(a, sigma, non_flat_curve["price"], 0.001)
    t = np.array([0, 0.5, 1, 5, 10, 30])
    expected = non_flat_curve["forward_derivative"](t) + a * non_flat_curve["forward"](t) + sigma**2 / (2 * a) * (1 - np.exp(-2 * a * t))
    assert np.max(np.abs(theta(t) - expected)) < 1e-8

def test_hull_white_mean_short_rate_on_non_flat_curve(non_flat_curve):
    # The mean of the short rate is alpha(t) = f(0,t) + sigma^2/(2 a^2) (1 - exp(-a t))^2, exactly
    # because of the moment matching, up to the finite difference approximation of f(0,t)
    a, sigma = 0.1, 0.01
    paths = calculate_hull_white_paths(100, 120, 10, non_flat_curve["price"], a, sigma, 0.001, rng=0)
    t = paths["time"]
    alpha = non_flat_curve["forward"](t) + sigma**2 / (2 * a**2) * (1 - np.exp(-a * t))**2
    assert np.max(np.abs(np.mean(paths["R"], axis=0) - alpha)) < 1e-8

def test_hull_white_theta_with_a_zero(zero_coupon_bond_prices):
    # a = 0 (Ho-Lee limit) is not supported and must raise an explicit error
    with pytest.raises(ValueError, match="must not be 0"):
        calculate_hull_white_theta(0.0, 0.01, zero_coupon_bond_prices, 0.001)

@pytest.mark.parametrize("parameters", [
    {"num_paths": 0},
    {"num_steps": 0},
    {"end_time": 0},
    {"mean_reversion_rate": 0.0},
    {"mean_reversion_rate": -0.1},
    {"volatility": -0.01},
    {"tolerance": 0.0},
])
def test_hull_white_paths_invalid_input(zero_coupon_bond_prices, parameters):
    # Invalid parameters must raise a ValueError before any simulation
    valid = {"num_paths": 10, "num_steps": 12, "end_time": 1, "mean_reversion_rate": 0.1, "volatility": 0.01, "tolerance": 0.01}
    valid.update(parameters)
    with pytest.raises(ValueError):
        calculate_hull_white_paths(function_zero_coupon_price=zero_coupon_bond_prices, **valid)

def test_hull_white_paths_zero_volatility_allowed(zero_coupon_bond_prices):
    # sigma = 0 is valid and gives a deterministic short rate
    np.random.seed(0)
    paths = calculate_hull_white_paths(10, 12, 1, zero_coupon_bond_prices, 0.1, 0.0, 0.01)
    assert np.allclose(paths["R"], paths["R"][0])
