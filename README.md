<div align="center">
  <a href="https://github.com/open-source-modelling" target="_blank">
    <picture>
      <img src="images/OSM_logo.jpeg" width=280 alt="Logo"/>
    </picture>
  </a>
</div>


<h1 align="center" style="border-botom: none">
  <b>
    🐍 Light Economic Generator 🐍     
  </b>
</h1>

</br>

The purpose of this repository is to create an open-source stochastic economic scenario generator using algorithms previously published by OSEM.
As a prototype (NOT TESTED!!!!), LEG supports 3 models:
 - [Black Sholes](https://github.com/open-source-modelling/insurance_python/tree/main/black_sholes)
 - [Vasicek](https://github.com/open-source-modelling/insurance_python/tree/main/vasicek_one_factor)
 - [Hull-White](https://github.com/open-source-modelling/insurance_python/tree/main/hull_white_one_factor)




The stochastic scenarios are available in two modelities. As an index (I) or as a discounting factor (D).

The example of the input:
|Calibration_ID	|model	|Type	|NoOfPaths	|NoOfSteps	|T	|a	|sigma	|epsilon	|Country	|selected_param_file	|selected_curves_file	|mu	|gamma|
|--|--|--| -----| ---| ---| ----| ----| ----| --------| ---------------| -----------------| ----| ---|
|11|HW|	I|	1000|	600|	50|	0.02|	0.02|	0.01|	Slovenia|	Param_no_VA.csv|	Curves_no_VA.csv|	   0|	  0|
|22|BS|	D|	1000|	600|	50|    0|	0.02|	0.01|	Slovenia|	Param_no_VA.csv|	Curves_no_VA.csv|	0.02|	  0|
|33| V|	D|	1000|	600|	50|	   0|	0.02|	0.01|	Slovenia|	Param_no_VA.csv|	Curves_no_VA.csv|	0.02|	0.3|


This prototype generates a csv that contains 1000 stochastic scenarios for each of the 3 rows and append it one bellow the other.

The script that starts the prototype is (also in main.py):

```python
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

combined_run.to_csv("Output/run.csv")
```
