""" Functions to compute liquidity costs on level 2 market data

"""
import pandas as pd
import numpy as np
from pathlib import Path
import etl
import sklearn
import config as cfg
from typing import List
from sklearn.impute import SimpleImputer
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from config import (
    COSTS_ASKS_IMAGE_FN,
    COSTS_BIDS_IMAGE_FN,
    COSTS_ASKS_R2_CSV_FN,
    COSTS_BIDS_R2_CSV_FN,
    COSTS_ASKS_CSV_FN,
    COSTS_BIDS_CSV_FN,
)

asks_image_path = cfg.FOLDER_PATHS_DICTS["FIGURES"] / COSTS_ASKS_IMAGE_FN
bids_image_path = cfg.FOLDER_PATHS_DICTS["FIGURES"] / COSTS_BIDS_IMAGE_FN
asks_csv_path = cfg.FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_ASKS_CSV_FN
bids_csv_path = cfg.FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_BIDS_CSV_FN
asks_r2_csv_path = cfg.FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_ASKS_R2_CSV_FN
bids_r2_csv_path = cfg.FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_BIDS_R2_CSV_FN

asks_cols_dict = {"x": "X_sum_quantity_Ask", "y": "S(X)_avg_price_share_Ask"}

bids_cols_dict = {"x": "X_sum_quantity_Bid", "y": "S(X)_avg_price_share_Bid"}


def compute_price_per_share(
    lob_df: pd.DataFrame, order_type: str, return_quantity_sum=True
) -> List[np.array]:
    """Compute average price per share from limit order book.
    
    Assumptions:
        Must have a levels in [0,99] (maximum two digit levels) for any single time event `t`. 
    
    References: 
        Calculation described in "Distilling Liquidity Costs from Limit Order Books"
        by Amaya, Rochen et al (2015).
        Paper source: https://www.sciencedirect.com/science/article/abs/pii/S0378426618301353
        
    :lob_df: Limit order book data.
    :order_type: Str - "Ask" or "Bid"
    :return_quantity_sum: Return the quantity sum for the bid type. For example sum of the volume of "Ask" levels.
    :return: Default returns S_average_price_per_share, X_sum_volume
    """
    assert order_type in ["Ask", "Bid"], "`order_type` must be in [`Ask`, `Bid]"
    temp = lob_df.copy(deep=True)
    temp.index = temp.Time_Minute

    # Follow the variable notation in 2018 paper to determine Liquidity Cost from the LOB
    for col in temp.columns:
        # Rename Price cols
        if f"{order_type}Price" in col:
            new_col = col.replace("L", "p")[0:3].replace("_", "")
            new_col = f"{new_col}_{order_type}"
            temp.rename({col: new_col}, axis=1, inplace=True)
        # Rename Size cols
        if f"{order_type}Size" in col:
            new_col = col.replace("L", "x")[0:3].replace("_", "")
            new_col = f"{new_col}_{order_type}"
            temp.rename({col: new_col}, axis=1, inplace=True)

    askprice_cols, askquantity_cols = (
        [f"p{i}_{order_type}" for i in range(1, 11)],
        [f"x{i}_{order_type}" for i in range(1, 11)],
    )
    prices, quantities = temp[askprice_cols].values, temp[askquantity_cols].values

    # Comutation:

    # (1) Numerator: Use Harmond Product for element wise multiplication across matrices
    #            then compute row-wise sum.
    temp["NUMERATOR_sum_Pi_dot_X_i"] = np.multiply(quantities, prices).sum(axis=1)
    # (2) Denominator: Compute row the sum of the quantities
    temp["DEMONINATOR_sum_Xi"] = temp[askquantity_cols].sum(axis=1)
    # (3) Compute average price per share
    S_average_price_per_share = (
        temp["NUMERATOR_sum_Pi_dot_X_i"] / temp["DEMONINATOR_sum_Xi"]
    )
    X_sum_volume = temp[askquantity_cols].sum(axis=1)
    if return_quantity_sum:
        return S_average_price_per_share.values, X_sum_volume.values
    else:
        return S_average_price_per_share.values


def get_processed_lob_time_series(lob_df) -> List[pd.DataFrame]:
    """ Compute processed Limit Order Book time series
    that will be used for regression analysis to determine 
    the liquidity cost.
    
    :lob_df: Limit Order Book input data
    :return: asks_regression_df, bids_regression_df
    """
    print("Commencing `get_processed_lob_time_series()` method")
    print("Computing price per share")
    S_ask, X_ask = compute_price_per_share(lob_df, "Ask")
    S_bid, X_bid = compute_price_per_share(lob_df, "Bid")

    lob_df[asks_cols_dict["x"]], lob_df[asks_cols_dict["y"]] = X_ask, S_ask
    lob_df[bids_cols_dict["x"]], lob_df[bids_cols_dict["y"]] = X_bid, S_bid

    lob_df.index = pd.to_datetime(lob_df["Time_Minute"])

    # Compute dataframes that will be used for regression
    # This method does some linear and mean based imputation
    print("Computing data for `Ask` price regression analysis")
    asks_regression_df = compute_regression_df(
        lob_df=lob_df,
        time_col="Time_Minute",
        sum_quantity_col=asks_cols_dict["x"],
        average_price_share_col=asks_cols_dict["y"],
    )

    print("Computing data for `Bid` price regression analysis")
    bids_regression_df = compute_regression_df(
        lob_df=lob_df,
        time_col="Time_Minute",
        sum_quantity_col=bids_cols_dict["x"],
        average_price_share_col=bids_cols_dict["y"],
    )
    print("Succesfully executed! Exiting function")
    print()
    return asks_regression_df, bids_regression_df


def compute_liquidity_cost(
    asks_regression_df: pd.DataFrame, bids_regression_df: pd.DataFrame, window: int = 88
) -> dict:
    """ Compute liquidity cost using rolling regression.
    
    Remark the first N=window_size values will be NaN. This is not an error, see StatsModels 
    documentation for further detail.
    
    :return: asks_rols_params, bids_rols_params
    """
    window_size = (
        24 * 2 * 2
    )  # Each row represents half an hour, build models over 2 day time horizon
    print("Computing liquidity costs from rolling regression analysis.")
    print(f"Window size = {window_size}")

    print("Computing liquidity costs for ask transactions")
    asks_rols_results, asks_rols_params = compute_rolling_regression(
        window_size=window_size,
        endog=asks_regression_df[asks_cols_dict["y"]],
        exog=asks_regression_df[asks_cols_dict["x"]],
    )

    print("Computing liquidity costs for bid transactions")
    bids_rols_results, bids_rols_params = compute_rolling_regression(
        window_size=window_size,
        endog=bids_regression_df[bids_cols_dict["y"]],
        exog=bids_regression_df[bids_cols_dict["x"]],
    )

    asks_rols_params.index = asks_regression_df["Time_Minute"]
    bids_rols_params.index = bids_regression_df["Time_Minute"]

    results_dict = {
        "asks_rols_results": asks_rols_results,
        "asks_rols_params": asks_rols_params,
        "bids_rols_results": bids_rols_results,
        "bids_rols_params": bids_rols_params,
    }

    # Get R2 score which explains variance of prediction
    asks_r2 = "{:.4f}".format(results_dict["asks_rols_results"].rsquared.mean())
    bids_r2 = "{:.4f}".format(results_dict["bids_rols_results"].rsquared.mean())
    print(
        f"R2 for asks transactions in Line Order Book (LOB) regression analysis: {asks_r2}"
    )
    print(
        f"R2 for bids transactions in Line Order Book (LOB) regression analysis: {bids_r2}"
    )
    print()
    print("Succesfully executed! Exiting function")
    print()
    return results_dict


def compute_regression_df(
    lob_df: pd.DataFrame,
    time_col: str,
    sum_quantity_col: str,
    average_price_share_col: str,
    freq: str = "30min",
    offset: str = "60min",
) -> pd.DataFrame:
    """ Function to impute regression DataFrame for Limit Order Book that will be used to calculate the liqudity cost
    over different intervals.
    
    :lob_df: Input LOB dataframe
    :col_names: List of column names 
    :time_col: Time column
    :sum_quantity_col: Column name containing sum of volume of derivative for example total sum of corporate bonds avaliable at time t
    :average_price_share_col: Column name containing average price per share for asset at time t
    """
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")

    lob_df.index = pd.to_datetime(lob_df[time_col])
    col_names = [sum_quantity_col, average_price_share_col]
    return_df = lob_df[col_names]
    return_df = return_df.groupby(
        pd.Grouper(freq=freq, offset=offset, label="right")
    ).mean()
    return_df.reset_index(inplace=True)

    return_df[sum_quantity_col].interpolate(method="linear", inplace=True)
    return_df[average_price_share_col].interpolate(method="linear", inplace=True)

    # Fill nulls/nans
    return_df[sum_quantity_col] = imp.fit_transform(
        return_df[sum_quantity_col].values.reshape(-1, 1)
    )
    return_df[average_price_share_col] = imp.fit_transform(
        return_df[average_price_share_col].values.reshape(-1, 1)
    )
    return return_df


# NOTE: THIS DID NOT WORK - DERIVATIVE ZERO ERROR
def train_bayesian_rolling_regressor(
    data: pd.DataFrame, dependent_col: str, response_col: str, tune: int = 500,
):
    """ Uses default configuration from: https://docs.pymc.io/notebooks/GLM-rolling-regression.html
    A simple rolling linear regression could have been used as well.
    :data: Input data
    :dependent_col: X column
    :response_col: Y column:
    :tune: Number of iterations to tune model
    """
    model_randomwalk = pm.Model()
    with model_randomwalk:
        # std of random walk
        sigma_alpha = pm.Exponential("sigma_alpha", 50.0)
        sigma_beta = pm.Exponential("sigma_beta", 50.0)

        alpha = pm.GaussianRandomWalk("alpha", sigma=sigma_alpha, shape=len(data))
        beta = pm.GaussianRandomWalk("beta", sigma=sigma_beta, shape=len(data))

    with model_randomwalk:
        # Define regression
        regression = alpha + beta * data[dependent_col]

        # Assume prices are normally distributed, the mean comes from the regression.
        sd = pm.HalfNormal("sd", sigma=0.1)
        likelihood = pm.Normal(
            "y", mu=regression, sigma=sd, observed=data[response_col]
        )

    with model_randomwalk:
        trace_rw = pm.sample(tune=500, cores=3, target_accept=0.9)
    return trace_rw


def compute_rolling_regression(
    window_size: int, endog: pd.DataFrame, exog: pd.DataFrame
):
    """ Wrapper function to compute rolling regression co-efficients 
    for pre-processed LOB using stats-models.
    
    Based on Amaya, Rochen et al (2015) we assume the coefficient is the liquidity cost
    and alpha is the intercept.
    
    Ref: 
        https://www.statsmodels.org/dev/examples/notebooks/generated/rolling_ls.html
        
        Calculation described in "Distilling Liquidity Costs from Limit Order Books"
        by Amaya, Rochen et al (2015).
        Paper source: https://www.sciencedirect.com/science/article/abs/pii/S0378426618301353
        
    :window_size: Size of the window
    :endog: Dependent variable - y
    :exog: Independent variable - x
    :return: rols_results (instance of statsmodels results object), rols_params (pd.DataFrame)
    """
    endog = endog
    exog = sm.add_constant(exog, prepend=False)
    rols = RollingOLS(endog, exog, window=window_size)
    rols_results = rols.fit()
    rols_params = rols_results.params
    rols_params.columns = ["liquidity_cost", "intercept"]
    return rols_results, rols_params


def save_liquidity_data(results_dict):
    """ Wrapper method to ave images and CSV data for linear regression analysis done using
    method compute_liquidity_costs for for ask and bid transactions on Limit Order Book.
    :results_dict: Results dictionary produced from compute_liquidity_cost() function.
    """

    def print_save_msg(path, file_type: str):
        print(f"Sucessfully saved {file_type} to: {path}")

    fig_asks_costs = results_dict["asks_rols_results"].plot_recursive_coefficient(
        variables=asks_cols_dict["x"], figsize=(14, 6)
    )
    fig_bids_costs = results_dict["bids_rols_results"].plot_recursive_coefficient(
        variables=bids_cols_dict["x"], figsize=(14, 6)
    )

    fig_asks_costs.savefig(asks_image_path, dpi=300)
    fig_bids_costs.savefig(bids_image_path, dpi=300)

    print_save_msg(asks_csv_path, "image")
    print_save_msg(bids_csv_path, "image")

    results_dict["asks_rols_params"].to_csv(asks_csv_path)
    results_dict["asks_rols_params"].to_csv(bids_csv_path)

    print_save_msg(asks_csv_path, "CSV")
    print_save_msg(bids_csv_path, "CSV")

    asks_r2 = pd.DataFrame(
        np.array(results_dict["asks_rols_results"].rsquared.mean().reshape(-1, 1)),
        columns=["asks_lob_regression_r2"],
    )
    bids_r2 = pd.DataFrame(
        np.array(results_dict["bids_rols_results"].rsquared.mean().reshape(-1, 1)),
        columns=["bids_lob_regression_r2"],
    )
    asks_r2.to_csv(asks_r2_csv_path)
    bids_r2.to_csv(bids_r2_csv_path)

    print_save_msg(asks_r2_csv_path, "CSV")
    print_save_msg(bids_r2_csv_path, "CSV")

    print("Succesfully executed! Terminating function.")


def run(save_data: bool = True) -> None:
    """ Entry-point function to run liquidity cost calculations and save image and static data 
    to files.
    """
    ticker_df, lob_df = etl.get_data()
    asks_regression_df, bids_regression_df = get_processed_lob_time_series(lob_df)
    results_dict = compute_liquidity_cost(asks_regression_df, bids_regression_df)
    if save_data:
        save_liquidity_data(results_dict)
    return


if __name__ == "__main__":
    run()
    print("Succesfully executed ! Terminating program.")
