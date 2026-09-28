import os
import pandas as pd
from read_input import read_model_input, validate_model_input, REPOSITORY_ROOT, DATA_FOLDER
from term_structure import calculate_zero_coupon_price
from black_scholes import set_up_black_scholes
from vasicek import set_up_vasicek
from hull_white import set_up_hull_white

param_raw = pd.read_csv(os.path.join(DATA_FOLDER, "Parameters.csv"), sep=',', index_col=0)

# All runs are checked before the first simulation starts.
validate_model_input(param_raw)

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