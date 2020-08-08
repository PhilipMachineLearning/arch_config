""" Config settings

"""
from pathlib import Path
import os

current_path = Path(os.path.abspath(__file__))

FOLDER_PATHS_DICTS = {
    "INPUT_DATA": current_path.parents[1] / "data" / "input",
    "STATIC_DATA": current_path.parents[1] / "data" / "static",
    "OUTPUT_DATA": current_path.parents[1] / "data" / "output",
    "CROSS_VALIDATION": current_path.parents[1] / "data" / "cross_validation",
    "FIGURES": current_path.parents[1] / "reports" / "figures"
}

INPUT_DICT = {
    "MINUTE_DATA_PATH": FOLDER_PATHS_DICTS["INPUT_DATA"] / "ABC_Level_One_Tick_Data.csv",
    "HOUR_DATA_PATH": FOLDER_PATHS_DICTS["INPUT_DATA"] / "ABC_Level_Two_Tick_Data.csv",
}


X_COLS = [
    "Time",
    "MinTimeHour",
    "MaxTimeHour",
    "Instrument_Code_id",
    "High",
    "Low",
    "VWAP",
    "TWAP",
    "NumberOfTrades",
    "Volume",
    "Turnover",
    "Avg_Bid_Ask_Spread",   
]

Y_COL = ["liquidity_cost"]

WINDOW_SIZE = 48  #
RANDOM_STATE = int(121301)
N_ESTIMATORS = [16, 32, 100, 150, 200, 250]
SCORING = "neg_mean_squared_error"
N_SPLITS = 10


# Set filenames (FN's) for liqiduity costs
COSTS_ASKS_IMAGE_FN = "lob_asks_liquidity_costs_regression_analysis.png" 
COSTS_BIDS_IMAGE_FN = "lob_bids_liquidity_costs_regression_analysis.png"
COSTS_ASKS_R2_CSV_FN = "lob_asks_r2_linear_regression.csv" 
COSTS_BIDS_R2_CSV_FN = "lob_bids_r2_linear_regression.csv"
COSTS_ASKS_CSV_FN = "lob_asks_liquidity_costs.csv" 
COSTS_BIDS_CSV_FN = "lob_bids_liquidity_costs.csv"
