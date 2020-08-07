""" Config settings

"""
from pathlib import Path

FOLDER_PATHS_DICTS = {
    "INPUT_DATA": Path.cwd().parents[0] / "data" / "input",
    "STATIC_DATA": Path.cwd().parents[0] / "data" / "static",
    "OUTPUT_DATA": Path.cwd().parents[0] / "data" / "output",
    "FIGURES":Path.cwd().parents[0] / "reports" / "figures"
}

INPUT_DICT = {
    "MINUTE_DATA_PATH": FOLDER_PATHS_DICTS["INPUT_DATA"] / "ABC_Level_One_Tick_Data.csv",
    "HOUR_DATA_PATH": FOLDER_PATHS_DICTS["INPUT_DATA"] / "ABC_Level_Two_Tick_Data.csv",
}

RANDOM_STATE = int(121301)
N_ESTIMATORS = [16, 32, 100, 150, 200, 250]


# Set filenames (FN's) for liqiduity costs
COSTS_ASKS_IMAGE_FN = "lob_asks_liquidity_costs_regression_analysis.png" 
COSTS_BIDS_IMAGE_FN = "lob_bids_liquidity_costs_regression_analysis.png"
COSTS_ASKS_R2_CSV_FN = "lob_asks_r2_linear_regression.csv" 
COSTS_BIDS_R2_CSV_FN = "lob_bids_r2_linear_regression.csv"
COSTS_ASKS_CSV_FN = "lob_asks_liquidity_costs.csv" 
COSTS_BIDS_CSV_FN = "lob_bids_liquidity_costs.csv"
