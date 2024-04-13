import pandas as pd
import numpy as np
from black_sholes import set_up_black_sholes
from vasicek import set_up_vasicek
from hull_white import set_up_hull_white
from term_structure import calculate_zero_coupon_price

param_raw = pd.read_csv("Parameters.csv", sep=',', index_col=0)

combined_run = []

for asset_id in param_raw.index:

    if param_raw["model"][asset_id] == "HW":
        run = set_up_hull_white(asset_id=asset_id)        
    elif param_raw["model"][asset_id] == "BS":
        run = set_up_black_sholes(asset_id=asset_id)
    elif param_raw["model"][asset_id] == "V":
        run = set_up_vasicek(asset_id=asset_id)
    else:
        raise ValueError("Model type not available")

    if isinstance(combined_run,pd.DataFrame):
        combined_run = pd.concat([combined_run,run])
    else:
        combined_run = run

combined_run.to_csv("Output/run.csv")


