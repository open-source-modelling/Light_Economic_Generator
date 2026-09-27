<div align="center">
  <a href="https://github.com/open-source-modelling" target="_blank">
    <picture>
      <img src="images/OSM_logo.jpeg" width=280 alt="Logo"/>
    </picture>
  </a>
</div>


<h1 align="center" style="border-bottom: none">
  <b>
    🐍 Light Economic Generator 🐍     
  </b>
</h1>

</br>

The purpose of this repository is to create an open-source stochastic economic scenario generator using algorithms previously published by Open-Source Modelling (OSM).

LEG is a prototype. The Hull-White and Black-Scholes models have automated tests and a validation notebook. The Vasicek model is not yet validated.

## Models

LEG supports 3 models. All of them use the EIOPA risk-free term structure, calculated with the Smith-Wilson algorithm, as input.

| Code | Model | What it simulates |
|--|--|--|
| `HW` | [Hull-White](https://github.com/open-source-modelling/insurance_python/tree/main/hull_white_one_factor) | Short rate $dr = (\theta(t) - a r) dt + \sigma dW$. The parameter $\theta(t)$ is fitted to the term structure, so the average simulated discount factor reproduces the input curve. |
| `V` | [Vasicek](https://github.com/open-source-modelling/insurance_python/tree/main/vasicek_one_factor) | Short rate $dr = \gamma (\mu - r) dt + \sigma dW$. Only the starting rate $r(0)$ is taken from the term structure, so the scenarios do not reproduce the input curve. |
| `BS` | [Black-Scholes](https://github.com/open-source-modelling/insurance_python/tree/main/black_sholes) | Equity index $dS = r(t) S dt + \sigma S dW$ with $S(0) = 1$. The drift $r(t)$ is the deterministic short rate implied by the term structure, so the discounted index is a martingale. |

## Output types

The stochastic scenarios are available in two modalities, as an index (I) or as a discount factor (D):

| Model | Index (I) | Discount factor (D) |
|--|--|--|
| Hull-White, Vasicek | Bank account $\exp\left(\int_0^t r(s) ds\right)$ | $\exp\left(-\int_0^t r(s) ds\right)$, different for every scenario |
| Black-Scholes | Equity index $S(t)$ | Zero-coupon price $P(0,t)$ from the term structure, the same for every scenario |

## Input

Each row of `Parameters.csv` specifies one run. The example input:

| Calibration_ID | model | Type | NoOfPaths | NoOfSteps | T | a | sigma | epsilon | Country | selected_param_file | selected_curves_file | mu | gamma |
|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 11 | HW | I | 10000 | 600 | 50 | 0.05 | 0.008 | 0.01 | Slovenia | Param_no_VA.csv | Curves_no_VA.csv | 0 | 0 |
| 22 | BS | I | 10000 | 600 | 50 | 0 | 0.2 | 0.01 | Slovenia | Param_no_VA.csv | Curves_no_VA.csv | 0 | 0 |
| 33 | V | D | 10000 | 600 | 50 | 0 | 0.02 | 0.01 | Slovenia | Param_no_VA.csv | Curves_no_VA.csv | 0.02 | 0.3 |

| Column | Description | Used by |
|--|--|--|
| `Calibration_ID` | Unique identifier of the run. | All |
| `model` | Model code: `HW`, `V` or `BS`. | All |
| `Type` | Output type: `I` (index) or `D` (discount factor). | All |
| `NoOfPaths` | Number of stochastic scenarios. | All |
| `NoOfSteps` | Number of time steps (Ex. 600 monthly steps for 50 years). | All |
| `T` | Time horizon in years. | All |
| `a` | Mean reversion speed. | HW |
| `sigma` | Volatility. | All |
| `epsilon` | Step size for the finite difference approximation of the instantaneous forward rate. | HW, V |
| `Country` | EIOPA curve to use. Must match a column name in the curve files. | All |
| `selected_param_file` | Smith-Wilson calibration published by EIOPA (observed maturities, calibration vector, UFR, alpha). | All |
| `selected_curves_file` | Spot rates published by EIOPA. | All |
| `mu` | Long-term mean of the short rate. | V |
| `gamma` | Mean reversion speed. | V |

## Output

The generator writes all scenarios into `Output/run.csv`, with the runs appended one below the other. Each row is one scenario, indexed by `Run` (model code and calibration id, Ex. `HW-11`) and `Scenario_number`. The columns are the time points in years, from 0 to T in steps of T/NoOfSteps.

The example input generates 10000 scenarios for each of the 3 rows. The resulting file is about 340 MB.

## Getting started

LEG requires Python with `numpy` and `pandas`. The validation notebooks also require `matplotlib` and Jupyter, and the tests require `pytest`.

The script that starts the prototype is (also in main.py):

```python
import os
import pandas as pd
from read_input import read_model_input
from term_structure import calculate_zero_coupon_price
from black_sholes import set_up_black_sholes
from vasicek import set_up_vasicek
from hull_white import set_up_hull_white

param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

combined_run = []

for run_id in param_raw.index:
    [modeling_parameters, curve_parameters] = read_model_input(run_id)
    
    zero_coupon_price = lambda t: calculate_zero_coupon_price(t, curve_parameters["target_maturities"], curve_parameters["calibration_vector"], curve_parameters["ultimate_forward_rate"], curve_parameters["convergence_speed"] )
    if modeling_parameters["run_type"] == "HW":
        run = set_up_hull_white(run_id, modeling_parameters, zero_coupon_price)        
    elif modeling_parameters["run_type"] == "BS":
        run = set_up_black_sholes(run_id, modeling_parameters, zero_coupon_price)
    elif modeling_parameters["run_type"] == "V":
        run = set_up_vasicek(run_id, modeling_parameters, zero_coupon_price)
    else:
        raise ValueError("Model type not available")

    if isinstance(combined_run,pd.DataFrame):
        combined_run = pd.concat([combined_run,run])
    else:
        combined_run = run

os.makedirs("Output", exist_ok=True)
combined_run.to_csv("Output/run.csv")
```

## Tests and validation

 - `test_HW.py` contains the unit tests for the forward rate, the Hull-White parameter $\theta(t)$ and the Hull-White simulation. Run them with `pytest`.
 - `VALIDATION BLACK SHOLES.ipynb` validates the Black-Scholes model: the term structure against the EIOPA published curve, deterministic checks, a martingale test and the distribution of the log return.
 - `VALIDATION HULL WHITE.ipynb` validates the Hull-White model.

## Other files

 - `all.py` contains the complete code in a single file, used as a source for GPT helpers.
 - `RAG/` contains the code as a notebook and as HTML, used for retrieval-augmented generation.
