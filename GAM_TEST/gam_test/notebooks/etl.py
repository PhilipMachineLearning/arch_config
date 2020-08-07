""" Utils for data ETL (Extraction Transformation and Loading)

"""
import pandas as pd
from pathlib import Path
from typing import List
from config import INPUT_DICT, FOLDER_PATHS_DICTS
import numpy as np
from config import (
    COSTS_ASKS_CSV_FN,
    COSTS_BIDS_CSV_FN,
)

def get_data(input_paths_dict=INPUT_DICT) -> List[pd.DataFrame]:
    """ Get input data.
    
    :return: ticker_df, limit_order_book_df
    """
    ticker_df = pd.read_csv(input_paths_dict["MINUTE_DATA_PATH"])
    limit_order_book_df = pd.read_csv(input_paths_dict["HOUR_DATA_PATH"])
    return ticker_df, limit_order_book_df

def get_costs_data(input_paths_dict=INPUT_DICT) -> "asks_costs_df, bids_costs_df":
    """ Get input costs data for bids and costs transactions on
    Limit Order Book (LOB) market data - level 2.
    
    :return: asks_costs_df, bids_costs_df
    """
    asks_costs_df = pd.read_csv(FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_ASKS_CSV_FN)
    bids_costs_df = pd.read_csv(FOLDER_PATHS_DICTS["STATIC_DATA"] / COSTS_BIDS_CSV_FN)
    return asks_costs_df, bids_costs_df



def denorm_order_book(
    limit_order_book: pd.DataFrame, trans_type: str = "Ask"
) -> pd.DataFrame:
    """ Dernomalize limit order book to standard form for bids.
    
    :limit_order_book: Level 2 market data as provided by data-provider.
    """
    assert trans_type in [
        "Ask",
        "Bid",
    ], "Transaction type `trans_type` must be `Ask` or `Bid`"
    # First compute impact cost from level two - order book data
    temp = limit_order_book_df.copy(deep=True)
    merge_cols = ["Time_Minute", "Level"]
    size_string, price_string = f"{trans_type}Size", f"{trans_type}Price"

    size_cols = [col for col in limit_order_book_df.columns if size_string in col]
    price_cols = [col for col in limit_order_book_df.columns if price_string in col]

    temp.index = temp["Time_Minute"]
    size_dict, price_dict = (
        {"level_1": "Level", 0: size_string},
        {"level_1": "Level", 0: "Price"},
    )

    # Unstack Size and Price
    size = temp[size_cols].stack().reset_index().rename(columns=size_dict)
    price = temp[price_cols].stack().reset_index().rename(columns=price_dict)

    # Pre-process
    for df in [size, price]:
        df["Level"] = df["Level"].apply(lambda row: row[0:2])

    final = price.merge(size, left_on=merge_cols, right_on=merge_cols)
    return final

