import pytest
import numpy as np
from term_structure import calculate_instantaneous_forward_rate

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
