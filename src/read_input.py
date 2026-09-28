import os
import difflib
import warnings
import pandas as pd
import numpy as np

# Root folder of the repository, one level above the folder "src".
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The input files (Parameters.csv and the files it refers to) are stored in the folder "data".
DATA_FOLDER = os.path.join(REPOSITORY_ROOT, "data")

def read_model_input(asset_id: int)->list:

    param_raw = pd.read_csv(os.path.join(DATA_FOLDER, "Parameters.csv"), sep=',', index_col=0)

    selected_param_file = param_raw["selected_param_file"][asset_id]
    selected_curves_file = param_raw["selected_curves_file"][asset_id]
    country = param_raw["Country"][asset_id]

    run_type = param_raw["model"][asset_id]
    num_paths = int(param_raw["NoOfPaths"][asset_id]) # Number of stochastic scenarios
    num_steps = int(param_raw["NoOfSteps"][asset_id]) # Number of equidistand discrete modelling points (50*12 = 600)
    end_time = param_raw["T"][asset_id]                 # Time horizon in years (A time horizon of 50 years; T=50)
    a =  param_raw["a"][asset_id]                # Hull-White mean reversion parameter a
    mu =  param_raw["mu"][asset_id]                # Vasicek long-term mean parameter mu
    sigma = param_raw["sigma"][asset_id]         # Volatility parameter sigma
    gamma = param_raw["gamma"][asset_id]         # Vasicek mean reversion speed parameter gamma
    tolerance =  param_raw["epsilon"][asset_id]     # Incremental distance used to calculate for numerical approximation
                    # of for example the instantaneous spot rate (Ex. 0.01 will use an interval 
                    # of 0.01 as a discreete approximation for a derivative)
    curve_type = param_raw["Type"][asset_id]
    # Seed of the random number generator. Without a seed (blank, or no column "seed"),
    # the scenarios are different in every run.
    seed = param_raw["seed"][asset_id] if "seed" in param_raw.columns else None
    seed = None if pd.isna(seed) else int(seed)

    param_raw = pd.read_csv(os.path.join(DATA_FOLDER, selected_param_file), sep=',', index_col=0)

    maturities_country_raw = param_raw.loc[:,country+"_Maturities"].iloc[6:]
    param_country_raw = param_raw.loc[:,country + "_Values"].iloc[6:]
    extra_param = param_raw.loc[:,country + "_Values"].iloc[:6]

    relevant_positions = pd.notna(maturities_country_raw.values)
    maturities_country = maturities_country_raw.iloc[relevant_positions]
    calibration_vector = param_country_raw.iloc[relevant_positions]
    curve_raw = pd.read_csv(os.path.join(DATA_FOLDER, selected_curves_file), sep=',',index_col=0)
    curve_country = curve_raw.loc[:,country]

    # Curve related parameters
    target_maturities = np.transpose(np.array(maturities_country.values))
    ultimate_forward_rate = extra_param.iloc[3]/100
    convergence_speed = extra_param.iloc[4]
    calibration_vector = np.transpose(np.array(calibration_vector.values))

    curve = {"target_maturities":target_maturities, "ultimate_forward_rate":ultimate_forward_rate, "calibration_vector":calibration_vector, "convergence_speed":convergence_speed}
    
    modeling_run ={"num_paths":num_paths, "num_steps":num_steps,"end_time":end_time, "mu":mu, "a":a, "sigma":sigma, "gamma": gamma, "tolerance":tolerance, "curve_type":curve_type, "run_type":run_type, "seed":seed}

    return [modeling_run, curve]


def validate_model_input(param_raw: pd.DataFrame) -> None:
    """
    Checks all run specifications of Parameters.csv before any scenario is simulated,
    and raises a single error that lists every problem found. It checks that:
     - all columns are present and every run has its own Calibration_ID,
     - the model is HW, BS or V and the output type is I or D,
     - the numeric parameters used by the model of the run are valid (see the column
       "Used by" in the README),
     - the seed is blank or a whole number of at least 0,
     - the input files exist in the folder "data" and contain the country,
     - all runs have the same time grid, because they are written into one table.
    Runs with the same seed use the same random numbers, so their scenarios are not
    independent. This is allowed, but gives a warning.

    Args:
        param_raw (DataFrame): content of Parameters.csv, indexed by Calibration_ID.

    Raises:
        ValueError: if any run specification is not valid.
    """
    required_columns = ["model", "Type", "NoOfPaths", "NoOfSteps", "T", "a", "sigma", "epsilon", "Country",
                        "selected_param_file", "selected_curves_file", "mu", "gamma"]
    missing_columns = [column for column in required_columns if column not in param_raw.columns]
    if missing_columns:
        raise ValueError("Parameters.csv is missing the columns: " + ", ".join(missing_columns))
    if param_raw.empty:
        raise ValueError("Parameters.csv does not contain any runs")

    # Numeric columns: the models that use them (None for all), the requirement and its check
    whole_number = lambda x: x >= 1 and x.is_integer()
    numeric_columns = [("NoOfPaths", None, "a whole number of at least 1", whole_number),
                       ("NoOfSteps", None, "a whole number of at least 1", whole_number),
                       ("T", None, "positive", lambda x: x > 0),
                       ("sigma", None, "at least 0", lambda x: x >= 0),
                       ("a", ["HW"], "positive", lambda x: x > 0),
                       ("mu", ["V"], "a number", lambda x: True),
                       ("gamma", ["V"], "positive", lambda x: x > 0),
                       ("epsilon", ["HW", "V"], "positive", lambda x: x > 0),
                       ("seed", None, "blank or a whole number of at least 0", lambda x: x >= 0 and x.is_integer())]

    errors = []
    if param_raw.index.hasnans:
        errors.append("Every run needs a Calibration_ID")
    for run_id in param_raw.index[param_raw.index.duplicated()].unique():
        errors.append(f"Calibration_ID {run_id} is used by more than one run")

    def describe(value):
        # Value as shown in an error message
        return "blank" if pd.isna(value) else repr(value)

    file_columns = {}
    def columns_in_file(file_name):
        # Column names of an input file in the folder "data", or None if the file does not exist
        if file_name not in file_columns:
            path = os.path.join(DATA_FOLDER, file_name)
            file_columns[file_name] = set(pd.read_csv(path, nrows=0, index_col=0).columns) if os.path.isfile(path) else None
        return file_columns[file_name]

    time_grids = {}
    runs_per_seed = {}
    for run_id, run in zip(param_raw.index, param_raw.to_dict("records")):
        model = run["model"]
        if model not in ["HW", "BS", "V"]:
            errors.append(f"Run {run_id}: model must be HW, BS or V, not {describe(model)}")
        if run["Type"] not in ["I", "D"]:
            errors.append(f"Run {run_id}: Type must be I (index) or D (discount factor), not {describe(run['Type'])}")

        numbers = {}
        for column, models, requirement, is_valid in numeric_columns:
            if column not in run or (models is not None and model not in models):
                continue
            if column == "seed" and pd.isna(run[column]):
                continue # A blank seed is allowed
            try:
                value = float(run[column])
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value) or not is_valid(value):
                errors.append(f"Run {run_id}: {column} must be {requirement}, not {describe(run[column])}")
            else:
                numbers[column] = value

        if "NoOfSteps" in numbers and "T" in numbers:
            time_grids[run_id] = (numbers["T"], int(numbers["NoOfSteps"]))
        if "seed" in numbers:
            runs_per_seed.setdefault(int(numbers["seed"]), []).append(run_id)

        country = run["Country"]
        if pd.isna(country):
            errors.append(f"Run {run_id}: Country is blank")
        for file_column in ["selected_param_file", "selected_curves_file"]:
            file_name = run[file_column]
            if pd.isna(file_name):
                errors.append(f"Run {run_id}: {file_column} is blank")
                continue
            file_name = str(file_name)
            columns = columns_in_file(file_name)
            if columns is None:
                errors.append(f"Run {run_id}: {file_column} {file_name!r} was not found in the folder data")
                continue
            if file_column == "selected_param_file":
                # The Smith-Wilson calibration has a column with maturities and a column with values per country
                countries = [name[:-len("_Values")] for name in columns
                             if name.endswith("_Values") and name[:-len("_Values")] + "_Maturities" in columns]
            else:
                countries = list(columns)
            if not pd.isna(country) and country not in countries:
                suggestion = difflib.get_close_matches(str(country), countries, n=1)
                hint = f". Did you mean {suggestion[0]!r}?" if suggestion else ""
                errors.append(f"Run {run_id}: Country {country!r} is not in {file_name}{hint}")

    if len(set(time_grids.values())) > 1:
        grids = "; ".join(f"run {run_id}: T = {end_time:g}, NoOfSteps = {num_steps}"
                          for run_id, (end_time, num_steps) in time_grids.items())
        errors.append(f"All runs are written into one table, so they must have the same T and NoOfSteps ({grids})")

    for seed, run_ids in runs_per_seed.items():
        if len(run_ids) > 1:
            warnings.warn(f"Runs {', '.join(str(run_id) for run_id in run_ids)} use the same seed {seed}, so they draw "
                          "the same random numbers and their scenarios are not independent", stacklevel=2)

    if errors:
        raise ValueError("Invalid input in Parameters.csv:\n - " + "\n - ".join(errors))
