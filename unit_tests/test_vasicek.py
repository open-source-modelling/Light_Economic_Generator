import math
import pytest
import numpy as np
import pandas as pd
from vasicek import calculate_vasicek_paths, set_up_vasicek

# Parameters of the Vasicek model used in the tests
MU = 0.02
SIGMA = 0.02
GAMMA = 0.3
TOLERANCE = 0.01
Z_SCORE_MAX = 4.5

# Define a simple zero-coupon bond price function for testing
@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.03 * np.asarray(t, dtype=float))  # Example yield curve: constant yield of 3%

@pytest.fixture
def paths(zero_coupon_bond_prices):
    np.random.seed(0)
    return calculate_vasicek_paths(10000, 600, 50, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)

def vasicek_bond_price(r, tau):
    # Closed-form price of a ZCB with time to maturity tau, given the short rate r
    B = (1 - np.exp(-GAMMA * tau)) / GAMMA
    ln_A = (B - tau) * (MU - SIGMA**2 / (2 * GAMMA**2)) - SIGMA**2 * B**2 / (4 * GAMMA)
    return np.exp(ln_A - B * r)

def z_score(samples, expected):
    # Deviation of the sample mean from the expected value, in standard errors
    return (np.mean(samples, axis=0) - expected) / (np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples)))

# Structure of the output

def test_vasicek_output_shapes(paths):
    for key in ["R", "M", "I"]:
        assert paths[key].shape == (10000, 601)
    assert np.allclose(paths["time"], np.linspace(0, 50, 601))

def test_vasicek_initial_values(paths):
    # The initial short rate is the forward rate of the curve, the discount factor starts at 1
    assert np.allclose(paths["R"][:, 0], 0.03, atol=1e-8)
    assert np.all(paths["M"][:, 0] == 1)
    assert np.allclose(paths["I"] * paths["M"], 1)

def test_vasicek_initial_short_rate_on_non_flat_curve(non_flat_curve):
    # r(0) is the forward rate f(0,0) of the curve, not a spot rate. It is calculated with a
    # finite difference with step epsilon, which can move it by about epsilon * df(0,0)/dt.
    epsilon = TOLERANCE
    paths = calculate_vasicek_paths(10, 12, 1, non_flat_curve["price"], MU, SIGMA, GAMMA, epsilon, rng=0)
    tolerance = 2 * epsilon * abs(non_flat_curve["forward_derivative"](0.0))
    assert np.allclose(paths["R"][:, 0], non_flat_curve["forward"](0.0), atol=tolerance, rtol=0)

def test_vasicek_single_path(zero_coupon_bond_prices):
    np.random.seed(0)
    single = calculate_vasicek_paths(1, 12, 1, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
    assert single["R"].shape == (1, 13)
    assert np.all(np.isfinite(single["R"]))

def test_vasicek_same_seed_same_output(zero_coupon_bond_prices):
    np.random.seed(1)
    first = calculate_vasicek_paths(100, 12, 1, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
    np.random.seed(1)
    second = calculate_vasicek_paths(100, 12, 1, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
    assert np.array_equal(first["R"], second["R"])

@pytest.mark.parametrize("curve_type, output", [("I", "I"), ("D", "M")])
def test_set_up_vasicek_output(zero_coupon_bond_prices, curve_type, output):
    modeling_parameters = {"num_paths": 50, "num_steps": 12, "end_time": 1, "mu": MU, "gamma": GAMMA,
                           "sigma": SIGMA, "tolerance": TOLERANCE, "curve_type": curve_type}
    np.random.seed(0)
    scenarios = set_up_vasicek(33, modeling_parameters, zero_coupon_bond_prices)
    np.random.seed(0)
    expected = calculate_vasicek_paths(50, 12, 1, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
    assert scenarios.index.names == ["Run", "Scenario_number"]
    assert set(scenarios.index.get_level_values("Run")) == {"V-33"}
    assert np.allclose(scenarios.columns.values.astype(float), expected["time"])
    assert np.allclose(scenarios.values, expected[output])

def test_set_up_vasicek_invalid_type(zero_coupon_bond_prices):
    modeling_parameters = {"num_paths": 10, "num_steps": 12, "end_time": 1, "mu": MU, "gamma": GAMMA,
                           "sigma": SIGMA, "tolerance": TOLERANCE, "curve_type": "X"}
    with pytest.raises(ValueError):
        set_up_vasicek(33, modeling_parameters, zero_coupon_bond_prices)

# Deterministic checks (sigma = 0)

def test_vasicek_zero_volatility_short_rate(zero_coupon_bond_prices):
    # Without noise, the short rate decays exponentially from r(0) towards mu
    deterministic = calculate_vasicek_paths(10, 600, 50, zero_coupon_bond_prices, MU, 0.0, GAMMA, TOLERANCE)
    t = deterministic["time"]
    r0 = deterministic["R"][0, 0]
    expected = MU + (r0 - MU) * np.exp(-GAMMA * t)
    assert np.max(np.abs(deterministic["R"] - expected)) < 1e-12

def test_vasicek_zero_volatility_discount_factor(zero_coupon_bond_prices):
    # Without noise, the discount factor is exactly exp(-integral of the short rate)
    deterministic = calculate_vasicek_paths(10, 600, 50, zero_coupon_bond_prices, MU, 0.0, GAMMA, TOLERANCE)
    t = deterministic["time"]
    r0 = deterministic["R"][0, 0]
    expected = np.exp(-(MU * t + (r0 - MU) * (1 - np.exp(-GAMMA * t)) / GAMMA))
    assert np.max(np.abs(deterministic["M"] / expected - 1)) < 1e-12

# Distribution of the short rate

def test_vasicek_mean_of_short_rate(paths):
    # Because of moment matching, the mean of the short rate is exact
    t = paths["time"]
    r0 = paths["R"][0, 0]
    expected = MU + (r0 - MU) * np.exp(-GAMMA * t)
    assert np.max(np.abs(np.mean(paths["R"], axis=0) - expected)) < 1e-12

def test_vasicek_variance_of_short_rate(paths):
    t = paths["time"][1:]
    n = paths["R"].shape[0]
    theoretical = SIGMA**2 / (2 * GAMMA) * (1 - np.exp(-2 * GAMMA * t))
    simulated = np.var(paths["R"][:, 1:], axis=0, ddof=1)
    z = (simulated - theoretical) / (theoretical * np.sqrt(2 / (n - 1)))
    assert np.max(np.abs(z)) < Z_SCORE_MAX

def test_vasicek_covariance_of_short_rate(paths):
    # Cov(r(s), r(t)) = exp(-gamma (t-s)) Var(r(s))
    R = paths["R"]
    n = R.shape[0]
    for s_index, t_index in [(12, 24), (120, 132), (360, 372)]:
        s, t = paths["time"][s_index], paths["time"][t_index]
        variance_s = SIGMA**2 / (2 * GAMMA) * (1 - np.exp(-2 * GAMMA * s))
        variance_t = SIGMA**2 / (2 * GAMMA) * (1 - np.exp(-2 * GAMMA * t))
        theoretical = np.exp(-GAMMA * (t - s)) * variance_s
        simulated = np.cov(R[:, s_index], R[:, t_index])[0, 1]
        # Standard error of the sample covariance of two normal variables
        standard_error = np.sqrt((variance_s * variance_t + theoretical**2) / n)
        assert abs(simulated - theoretical) / standard_error < Z_SCORE_MAX

def test_vasicek_distribution_does_not_depend_on_time_step(zero_coupon_bond_prices):
    # The transition of the short rate is exact, so the distribution at T is the same
    # for any number of steps
    theoretical_variance = SIGMA**2 / (2 * GAMMA) * (1 - np.exp(-2 * GAMMA * 10))
    for num_steps in [5, 500]:
        np.random.seed(0)
        coarse = calculate_vasicek_paths(20000, num_steps, 10, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
        r_T = coarse["R"][:, -1]
        r0 = coarse["R"][0, 0]
        assert abs(np.mean(r_T) - (MU + (r0 - MU) * np.exp(-GAMMA * 10))) < 1e-12
        z = (np.var(r_T, ddof=1) - theoretical_variance) / (theoretical_variance * np.sqrt(2 / 19999))
        assert abs(z) < Z_SCORE_MAX

# Prices compared to the closed-form Vasicek prices (not the input term structure)

def test_vasicek_discount_factor_matches_closed_form(paths):
    r0 = paths["R"][0, 0]
    for index in [12, 120, 360, 600]:
        expected = vasicek_bond_price(r0, paths["time"][index])
        assert abs(z_score(paths["M"][:, index], expected)) < Z_SCORE_MAX

def test_vasicek_future_bond_price_matches_closed_form(paths):
    # E[D(S) P(S,T)] = P(0,T)
    r0 = paths["R"][0, 0]
    for S in [1, 5, 10, 20, 30]:
        index = S * 12
        discounted_bond = paths["M"][:, index] * vasicek_bond_price(paths["R"][:, index], 10)
        assert abs(z_score(discounted_bond, vasicek_bond_price(r0, S + 10))) < Z_SCORE_MAX

def test_vasicek_bond_option_matches_closed_form(paths):
    # At-the-money call option with expiry S on a ZCB maturing at T = S + 10
    normal_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    r0 = paths["R"][0, 0]
    for S in [1, 5, 10, 20, 30]:
        T = S + 10
        index = S * 12
        P0S, P0T = vasicek_bond_price(r0, S), vasicek_bond_price(r0, T)
        strike = P0T / P0S
        sigma_p = SIGMA / GAMMA * (1 - np.exp(-GAMMA * (T - S))) * np.sqrt((1 - np.exp(-2 * GAMMA * S)) / (2 * GAMMA))
        h = np.log(P0T / (P0S * strike)) / sigma_p + sigma_p / 2
        closed_form = P0T * normal_cdf(h) - strike * P0S * normal_cdf(h - sigma_p)
        payoff = paths["M"][:, index] * np.maximum(vasicek_bond_price(paths["R"][:, index], T - S) - strike, 0)
        assert abs(z_score(payoff, closed_form)) < Z_SCORE_MAX

def test_vasicek_does_not_reproduce_input_curve(paths, zero_coupon_bond_prices):
    # Vasicek only takes r(0) from the term structure. With mu below the curve, the
    # discount factors are far above the input curve. This is expected behaviour and
    # must not be turned into a martingale test.
    assert z_score(paths["M"][:, 600], zero_coupon_bond_prices(50.0)) > 10

# Input validation

def test_vasicek_gamma_zero_raises(zero_coupon_bond_prices):
    # gamma is read from the csv as a numpy float, which previously produced NaN
    with pytest.raises(ValueError, match="must not be 0"):
        calculate_vasicek_paths(10, 12, 1, zero_coupon_bond_prices, MU, SIGMA, np.float64(0.0), TOLERANCE)

@pytest.mark.parametrize("parameters", [
    {"num_paths": 0},
    {"num_steps": 0},
    {"end_time": 0},
    {"gamma": -0.3},
    {"sigma": -0.02},
    {"tolerance": 0.0},
])
def test_vasicek_paths_invalid_input(zero_coupon_bond_prices, parameters):
    valid = {"num_paths": 10, "num_steps": 12, "end_time": 1, "mean_drift": MU, "sigma": SIGMA, "gamma": GAMMA, "tolerance": TOLERANCE}
    valid.update(parameters)
    with pytest.raises(ValueError):
        calculate_vasicek_paths(function_zero_coupon_price=zero_coupon_bond_prices, **valid)

def test_vasicek_discount_factor_does_not_depend_on_time_step(zero_coupon_bond_prices):
    # The discount factor is sampled exactly, so its expectation matches the closed-form
    # bond price for any number of steps
    for num_steps in [5, 500]:
        np.random.seed(0)
        coarse = calculate_vasicek_paths(20000, num_steps, 10, zero_coupon_bond_prices, MU, SIGMA, GAMMA, TOLERANCE)
        expected = vasicek_bond_price(coarse["R"][0, 0], 10)
        assert abs(z_score(coarse["M"][:, -1], expected)) < Z_SCORE_MAX
