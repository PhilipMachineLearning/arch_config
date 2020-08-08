''' Utils for feature engineering

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from tsfel.utils.signal_processing import signal_window_splitter
import tsfel
import etl

TIME_COLS = ["Time", "MinTimeHour", "MaxTimeHour"]


def get_time_cols(input_df:pd.DataFrame, time_cols:pd.DataFrame, sin_cos_transform:bool=False):
    """ Get time cols with the sin/cos of time cols.
    computed. 
    
    Remark: This transformation will not work on categoric time features.
    For example a one-hot-encoding column which represnts day of the 
    week.
    
    Sources:
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        https://stats.stackexchange.com/questions/311494/best-practice-for-encoding-datetime
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#from-timestamps-to-epoch 
        
    :input_df: Input
    :time_cols: Time columns 
    :return: return_df, new_time_cols
    """
    def datetime_to_integer(time_df):
        return (time_df - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')  
        
    return_df = input_df.copy(deep=True)
    seconds_in_day = 24*60*60
    new_time_cols = []
    
    for col in time_cols:
        return_df[col] = pd.to_datetime(return_df[col])
        return_df[col] = return_df[col].dt.tz_localize(None)
        
        # Convert from timestamp to UNIX EPOCH
        return_df[col] = datetime_to_integer(return_df[col])
        
        if sin_cos_transform:
            # Apply sin/cos transformations
            sin_col, cos_col = f'{col}_Sin', f'{col}_Cos'
            return_df[sin_col] = np.sin(2*np.pi*return_df[col]/seconds_in_day)
            return_df[cos_col] = np.cos(2*np.pi*return_df[col]/seconds_in_day)
            # store time cols
            new_time_cols.extend([col, sin_col,cos_col] )
        else:
            new_time_cols.extend([col])
        
    return return_df, new_time_cols


def get_data(sin_cos_transform=False) -> "asks_merged_df, bids_merged_df":
    """ Get data for ML modelling
    :return: asks_merged_df, bids_merged_df
    """
    # Get data
    ticker_df, lob_df = etl.get_data()
    asks_costs_df, bids_costs_df = etl.get_costs_data()
    asks_costs_df.dropna(axis=0, inplace=True)
    bids_costs_df.dropna(axis=0, inplace=True)

    ticker_df.rename({"Time_Hour": "Time"}, axis=1, inplace=True)
    for df in [asks_costs_df, bids_costs_df]:
        df.rename({"Time_Minute": "Time"}, axis=1, inplace=True)

    for df in [ticker_df, asks_costs_df, bids_costs_df]:
        df["Time"] = pd.to_datetime(df["Time"], utc=True)


    asks_merged_df = ticker_df.merge(asks_costs_df, on="Time")
    bids_merged_df = ticker_df.merge(asks_costs_df, on="Time")

    asks_merged_df, asks_new_time_cols = get_time_cols(
        input_df=asks_merged_df, time_cols=TIME_COLS, sin_cos_transform=sin_cos_transform
    )
    bids_merged_df, bids_new_time_cols = get_time_cols(
        input_df=bids_merged_df, time_cols=TIME_COLS, sin_cos_transform=sin_cos_transform
    )

    asks_merged_df["Instrument_Code"] = asks_merged_df["Instrument_Code"].astype("category")
    bids_merged_df["Instrument_Code"] = asks_merged_df["Instrument_Code"].astype("category")

    asks_merged_df["Instrument_Code_id"] = asks_merged_df["Instrument_Code"].cat.codes
    bids_merged_df["Instrument_Code_id"] = bids_merged_df["Instrument_Code"].cat.codes
    return asks_merged_df, bids_merged_df


def split_by_window_size(y:pd.DataFrame, window_size:int=5) -> pd.DataFrame:
    """ Wrapper function to get predictor variable values
    :y: Predictor variable
    :window_size: Size of the window
    """
    y_windows = signal_window_splitter(y, window_size=window_size)
    y = []
    for window in y_windows:
        y.append(window.iloc[-1])
    y = pd.DataFrame(y).reset_index(drop=True)
    return y

def auto_feature_engineering(input_df:pd.DataFrame,cols:List[str], window_size:int=5) -> pd.DataFrame:
    """ Automated feature engineering wrapper using TSFEL 
    package for features generated based on temporal metrics
    
    :input_df: Input data
    :window_size: Size of the window
    :return: X data
    """
    cfg_file = tsfel.get_features_by_domain("temporal") 
    X_df_list = []
    for col in cols:
        current_df = tsfel.time_series_features_extractor(
            cfg_file,
            input_df, 
            fs=50,
            window_splitter=True, 
            window_size=window_size,
        )
        current_df = current_df.add_prefix(col)
        X_df_list.append(current_df)

    X = pd.concat(X_df_list, axis=1)
    return X

