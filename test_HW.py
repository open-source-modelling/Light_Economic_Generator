import pytest
import numpy as np
from term_structure import *
from HW_run import *

# Define a simple zero-coupon bond price function for testing
@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.05 * t)  # Example yield curve: constant yield of 5%

def test_f0t_at_t0(zero_coupon_bond_prices):
    # Test the value of f0t at t = 0
    f0t_value = f0t(0, zero_coupon_bond_prices, 0.001)
    expected_value = 0.05  # For a constant yield curve, the forward rate equals the yield
    assert pytest.approx(f0t_value, abs=1e-4) == expected_value

def test_f0t_at_t_positive(zero_coupon_bond_prices):
    # Test the value of f0t at a positive time
    t = 1
    f0t_value = f0t(t, zero_coupon_bond_prices, 0.001)
    expected_value = -np.log(zero_coupon_bond_prices(t)) / t
    assert pytest.approx(f0t_value, abs=1e-4) == expected_value

def test_f0t_with_epsilon_zero(zero_coupon_bond_prices):
    # Test behavior when epsilon is zero
    with pytest.raises(ValueError):
        f0t(1, zero_coupon_bond_prices, 0)

@pytest.fixture
def zero_coupon_bond_prices():
    return lambda t: np.exp(-0.05 * t)  # Example yield curve: constant yield of 5%

def test_HW_theta_at_t0(zero_coupon_bond_prices):
    # Test the value of HW_theta at t = 0
    hw_theta = HW_theta(0.1, 0.01, zero_coupon_bond_prices, 0.001)
    hw_theta_value = hw_theta(0)
    expected_value = 0.01  # Expected theta value
    assert pytest.approx(hw_theta_value, abs=1e-4) == expected_value

def test_HW_theta_at_t_positive(zero_coupon_bond_prices):
    # Test the value of HW_theta at a positive time
    hw_theta = HW_theta(0.1, 0.01, zero_coupon_bond_prices, 0.001)
    hw_theta_value = hw_theta(1)
    expected_value = 0.1 * -np.log(zero_coupon_bond_prices(1)) / 1 + \
                     0.01**2 / (2 * 0.1) * (1 - np.exp(-2 * 0.1 * 1))
    assert pytest.approx(hw_theta_value, abs=1e-4) == expected_value