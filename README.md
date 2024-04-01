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
 - Black Sholes
 - Vasicek
 - Hull-White

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
from main_BS import Run_Black_Sholes
from main_V import Run_Vasicek
from main_HW import Run_Hull_White

param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)


combined_run = []

for asset_id in param_raw.index:
    if param_raw["model"][asset_id] == "HW":
        print("HW")
        run = Run_Hull_White(asset_id=asset_id)        
    elif param_raw["model"][asset_id] == "BS":
        print("BS")
        run = Run_Black_Sholes(asset_id=asset_id)
    elif param_raw["model"][asset_id] == "V":
        print("V")
        run = Run_Vasicek(asset_id=asset_id)
    else:
        print("Model not available")

    if isinstance(combined_run,pd.DataFrame):
        combined_run = pd.concat([combined_run,run])
    else:
        combined_run = run

combined_run.to_csv("Output/run.csv")
```
