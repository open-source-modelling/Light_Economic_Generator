import shutil
import pytest
import numpy as np
import read_input

@pytest.fixture
def data_folder(tmp_path, monkeypatch):
    # Copy of the folder "data", so that a test can change Parameters.csv
    shutil.copytree(read_input.DATA_FOLDER, tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(read_input, "DATA_FOLDER", str(tmp_path))
    return tmp_path

@pytest.fixture
def non_flat_curve():
    """
    Non-flat term structure with closed-form forward rates (Nelson-Siegel), rising from
    3% at t = 0 to 4% in the long end:

        f(0,t) = b0 + b1 exp(-t/tau) + b2 t/tau exp(-t/tau)

    On a flat curve, the forward rate equals the spot rate and its derivative is 0, which
    hides errors in both. The price P(0,t) = exp(-integral of f over [0, t]) is defined
    for any t, including the negative times used by centered finite differences at t = 0.
    """
    b0, b1, b2, tau = 0.04, -0.01, 0.005, 2.0

    def price(t):
        t = np.asarray(t, dtype=float)
        decay = np.exp(-t / tau)
        return np.exp(-(b0 * t + b1 * tau * (1 - decay) + b2 * (tau * (1 - decay) - t * decay)))

    def forward(t):
        t = np.asarray(t, dtype=float)
        return b0 + (b1 + b2 * t / tau) * np.exp(-t / tau)

    def forward_derivative(t):
        t = np.asarray(t, dtype=float)
        return (b2 * (1 - t / tau) - b1) / tau * np.exp(-t / tau)

    return {"price": price, "forward": forward, "forward_derivative": forward_derivative}
