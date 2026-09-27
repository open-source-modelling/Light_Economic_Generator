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

LEG is a prototype. The Hull-White and Vasicek models have automated tests, and all three models have a validation notebook.

## Models

LEG supports 3 models. All of them use the EIOPA risk-free term structure, calculated with the Smith-Wilson algorithm, as input.

| Code | Model | What it simulates |
|--|--|--|
| `HW` | [Hull-White](https://github.com/open-source-modelling/insurance_python/tree/main/hull_white_one_factor) | Short rate $dr = (\theta(t) - a r) dt + \sigma dW$. The parameter $\theta(t)$ is fitted to the term structure, so the average simulated discount factor reproduces the input curve. |
| `V` | [Vasicek](https://github.com/open-source-modelling/insurance_python/tree/main/vasicek_one_factor) | Short rate $dr = \gamma (\mu - r) dt + \sigma dW$. Only the starting rate $r(0)$ is taken from the term structure, so the scenarios do not reproduce the input curve. With the example parameters, the Vasicek bond prices are up to 100 bps of yield away from the input curve. |
| `BS` | [Black-Scholes](https://github.com/open-source-modelling/insurance_python/tree/main/black_sholes) | Equity index $dS = r(t) S dt + \sigma S dW$ with $S(0) = 1$. The drift $r(t)$ is the deterministic short rate implied by the term structure, so the discounted index is a martingale. |

## Output types

The stochastic scenarios are available in two modalities, as an index (I) or as a discount factor (D):

| Model | Index (I) | Discount factor (D) |
|--|--|--|
| Hull-White, Vasicek | Bank account $\exp\left(\int_0^t r(s) ds\right)$ | $\exp\left(-\int_0^t r(s) ds\right)$, different for every scenario |
| Black-Scholes | Equity index $S(t)$ | Zero-coupon price $P(0,t)$ from the term structure, the same for every scenario |

## Input

Each row of `data/Parameters.csv` specifies one run. The input files it refers to (`selected_param_file` and `selected_curves_file`) are also stored in the folder `data`. The example input:

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

The generator writes all scenarios into `output/run.csv` in the root folder of the repository, with the runs appended one below the other. Each row is one scenario, indexed by `Run` (model code and calibration id, Ex. `HW-11`) and `Scenario_number`. The columns are the time points in years, from 0 to T in steps of T/NoOfSteps.

The example input generates 10000 scenarios for each of the 3 rows. The resulting file is about 340 MB.

## Folder structure

```
Light_Economic_Generator/
├── src/                  Model code and the script that runs the generator
├── notebooks/            Validation notebooks
├── unit_tests/           Unit tests
├── data/                 Input files
│   ├── Parameters.csv        Run specifications
│   ├── Param_no_VA.csv       Smith-Wilson calibration published by EIOPA
│   └── Curves_no_VA.csv      Spot rates published by EIOPA
├── output/               Generated scenarios (created by the generator, not in git)
└── pytest.ini            Test configuration (tests in unit_tests, code in src)
```

## Getting started

LEG requires Python with `numpy` and `pandas`. The validation notebooks also require `matplotlib` and Jupyter, and the tests require `pytest`.

Run the generator from the root folder of the repository with:

```
python src/main.py
```

The input files are read from the folder `data` and the output is written to the folder `output`. Both paths are relative to the root folder of the repository, so the script can also be started from any other folder.

The script that starts the prototype is (`src/main.py`):

```python
import os
import pandas as pd
from read_input import read_model_input, REPOSITORY_ROOT, DATA_FOLDER
from term_structure import calculate_zero_coupon_price
from black_scholes import set_up_black_scholes
from vasicek import set_up_vasicek
from hull_white import set_up_hull_white

param_raw = pd.read_csv(os.path.join(DATA_FOLDER, "Parameters.csv"), sep=',', index_col=0)

combined_run = []

for run_id in param_raw.index:
    [modeling_parameters, curve_parameters] = read_model_input(run_id)
    
    zero_coupon_price = lambda t: calculate_zero_coupon_price(t, curve_parameters["target_maturities"], curve_parameters["calibration_vector"], curve_parameters["ultimate_forward_rate"], curve_parameters["convergence_speed"] )
    if modeling_parameters["run_type"] == "HW":
        run = set_up_hull_white(run_id, modeling_parameters, zero_coupon_price)        
    elif modeling_parameters["run_type"] == "BS":
        run = set_up_black_scholes(run_id, modeling_parameters, zero_coupon_price)
    elif modeling_parameters["run_type"] == "V":
        run = set_up_vasicek(run_id, modeling_parameters, zero_coupon_price)
    else:
        raise ValueError("Model type not available")

    if isinstance(combined_run,pd.DataFrame):
        combined_run = pd.concat([combined_run,run])
    else:
        combined_run = run

# The output is written to the folder "output" in the root folder of the repository.
output_folder = os.path.join(REPOSITORY_ROOT, "output")
os.makedirs(output_folder, exist_ok=True)
combined_run.to_csv(os.path.join(output_folder, "run.csv"))
```

## Tests and validation

The unit tests are in the folder `unit_tests`:

 - `test_term_structure.py` contains the unit tests for the instantaneous forward rate.
 - `test_hull_white.py` contains the unit tests for the Hull-White parameter $\theta(t)$ and the Hull-White simulation, including input validation.
 - `test_vasicek.py` contains the unit tests for the Vasicek simulation: the output structure, deterministic checks, the distribution of the short rate, closed-form prices of bonds and bond options, and input validation.

Run all tests from the root folder of the repository with `pytest`.

The validation notebooks are in the folder `notebooks`. They read the input files from the folder `data` and import the model code from the folder `src`:

 - `validation_black_scholes.ipynb` validates the Black-Scholes model: the term structure against the EIOPA published curve, deterministic checks, a martingale test and the distribution of the log return.
 - `validation_hull_white.ipynb` validates the Hull-White model: the term structure against the EIOPA published curve, deterministic checks, a martingale test, the distribution of the short rate and closed-form prices of bonds and bond options.
 - `validation_vasicek.ipynb` validates the Vasicek model: the term structure against the EIOPA published curve, deterministic checks, closed-form bond prices, the distribution of the short rate and closed-form prices of future bonds and bond options. It also shows how far the Vasicek bond prices are from the input curve.
