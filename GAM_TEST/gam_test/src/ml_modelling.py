""" Machine Learning Utils for modelling and prediction.

Sources: 
    http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
    https://robjhyndman.com/hyndsight/tscv/
"""
import config as cfg
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from ngboost import NGBRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from config import RANDOM_STATE, N_ESTIMATORS, X_COLS, Y_COL, WINDOW_SIZE
import feature_engineering as fe
import tsfel
import time
from tscv import GapKFold

tscv = TimeSeriesSplit(n_splits=cfg.N_SPLITS)


MODELS = {
    # Train in order of computational complexity
    "RandomForestRegressor": RandomForestRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor(),
    "BaggingRegressor": BaggingRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor(),
    "NGBRegressor":NGBRegressor(verbose=False),
    #"SVR": SVR(), #SVR too slow on local machine high space/time complexity
    "catboostregressor": CatBoostRegressor(verbose=False),
}

PARAMS = {
    "RandomForestRegressor": {
        "n_estimators": N_ESTIMATORS,
        "random_state": [RANDOM_STATE],
    },
    "AdaBoostRegressor": {"n_estimators": N_ESTIMATORS, "random_state": [RANDOM_STATE]},
    "BaggingRegressor": {"n_estimators": N_ESTIMATORS, "random_state": [RANDOM_STATE],},
    "ExtraTreesRegressor": {
        "n_estimators": N_ESTIMATORS,
        "random_state": [RANDOM_STATE],
        "max_depth": [1, 2, 3, 4, 5, 10, None],
    },
    "catboostregressor": {
        "n_estimators": N_ESTIMATORS,
        "random_seed": [RANDOM_STATE],
        "depth": [1, 2, 3, 4, 5, 10, None],
    },
    "NGBRegressor": {
        "n_estimators": N_ESTIMATORS,
        "random_state": [RANDOM_STATE],
    },
    "SVR": [
        # Kernel transforms data to new plane, C: Regularization parameter
        {"kernel": ["linear", "rbf", "sigmoid"]}, #"C": [1, 10]},
    ],
}


def get_X_y(input_df: pd.DataFrame, target_col: str) -> "X, y":
    """ Get input matrix X and predictor variable.
    :target_col: Name of target column
    :return: X, y
    """
    y = target_col
    X, y = input_df.drop(y, axis=1), input_df[y]
    return X, y


class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=tscv, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(
                model,
                params,
                #cv=cv,
                cv =cv, # Time series cross validation required
                n_jobs=n_jobs,
                verbose=verbose,
                scoring=scoring,
                refit=refit,
                return_train_score=True,
            )
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by="mean_score"):
        def row(key, scores, params):
            d = {
                "estimator": key,
                "min_score": min(scores),
                "max_score": max(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            params = self.grid_searches[k].cv_results_["params"]
            scores = []
            if isinstance(self.grid_searches[k], sklearn.model_selection._split.TimeSeriesSplit):
                splits = [key for key in self.grid_searches[k].cv_results_.keys() if "split" in key]
                for key in splits:
                    r = self.grid_searches[k].cv_results_[key]       
                    scores.append(r.reshape(len(params),1))

            else:
                for i in range(self.grid_searches[k].cv):
                    key = "split{}_test_score".format(i)
                    r = self.grid_searches[k].cv_results_[key]
                    scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p))) 

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ["estimator", "min_score", "mean_score", "max_score", "std_score"]
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

    
def get_cv_scores(X:pd.DataFrame, y:pd.DataFrame, scoring='neg_mean_squared_error'):
    """ Cross-validation prediction
    :X: Input
    :y: Predictor variable
    :scoring: Scoring metric
    """
    cv_helper = EstimatorSelectionHelper(
        MODELS, 
        PARAMS,
    )
    cv_helper.fit(X, y,cv=5, scoring='neg_mean_squared_error')
    
    scores = cv_helper.score_summary(sort_by='max_score')
    scores_col = [col for col in scores.columns if "score" in col]
    for col in scores_col:
        # Turn neg_mean_squared_error to MSE
        scores[col] = scores[col] * -1
    return scores

def get_cv_results(merged_df, freq:str="daily", auto_feature_extraction:bool=False, drop_correlated:bool=False):
    """ Wrapper function to get all Cross-Validation Scores
    """
    assert freq in ["30min", "daily"], "Frequency must be `30min` or `daily"
    merged_df[Y_COL] = merged_df[Y_COL].values.ravel()
    if freq == "daily":
        if not auto_feature_extraction:
            X = fe.split_by_window_size(merged_df[X_COLS], window_size=WINDOW_SIZE)
            y = fe.split_by_window_size(merged_df[Y_COL], window_size=WINDOW_SIZE)
            X["horizon_days"] = np.array(range(0, len(X)))

            if drop_correlated:
                # Drop Highly correlated features 
                corr_features = tsfel.correlated_features(X)
                X.drop(corr_features, axis=1, inplace=True)

            scores = get_cv_scores(X,y)
        else:
            X_auto_fe = fe.auto_feature_engineering(merged_df[X_COLS],cols=X_COLS, window_size=WINDOW_SIZE)
            y = fe.split_by_window_size(merged_df[Y_COL], window_size=WINDOW_SIZE)
            X_auto_fe["horizon_days"] = np.array(range(0, len(X_auto_fe)))

            if drop_correlated:
                # Drop Highly correlated features 
                corr_features = tsfel.correlated_features(X_auto_fe)
                X_auto_fe.drop(corr_features, axis=1, inplace=True)

            scores = get_cv_scores(X_auto_fe,y)
            
    if freq == "30min":
        X, y = get_X_y(merged_df, "liquidity_cost")
        X = X[X_COLS]
        X["horizon_days"] = np.array(range(0, len(X)))
        if drop_correlated:
            # Drop Highly correlated features 
            corr_features = tsfel.correlated_features(X)
            X.drop(corr_features, axis=1, inplace=True)
        
        scores = get_cv_scores(X,y)
    return scores

def get_scores_daily_models(asks_merged_df, bids_merged_df):
    """ Get all scores for model selection
    
    :asks_merged_df: Preprocessed asks data (merge of level 1 and 2 market data)
    :bids_merged_df: Preprocessed bids (merge of level 1 and 2 market data)
    :return: Dictionary containing scores
    """
    bids_scores_false_false = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=False, 
        drop_correlated=False,
        freq="daily",
    )

    bids_scores_false_true = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=False, 
        drop_correlated=True,
        freq="daily",
    )

    bids_scores_true_true = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=True, 
        drop_correlated=True,
        freq="daily",
    )

    bids_scores_true_false = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=True, 
        drop_correlated=False,
        freq="daily",
    )
    
    ask_scores_false_false = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=False, 
        drop_correlated=False,
        freq="daily",
    )

    ask_scores_false_true = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=False, 
        drop_correlated=True,
        freq="daily",
    )

    ask_scores_true_true = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=True, 
        drop_correlated=True,
        freq="daily",
    )

    ask_scores_true_false = get_cv_results(
        asks_merged_df,
        auto_feature_extraction=True, 
        drop_correlated=False,
        freq="daily",
    )
    return_dict = {
        "bids_scores_false_false":bids_scores_false_false,
        "bids_scores_false_true":bids_scores_false_true,
        "bids_scores_true_true":bids_scores_true_true,
        "bids_scores_true_false":bids_scores_true_false,
        "ask_scores_false_false":ask_scores_false_false,
        "ask_scores_false_true":ask_scores_false_true,
        "ask_scores_true_true":ask_scores_true_true,
        "ask_scores_true_false":ask_scores_true_false,
        
    }
    return return_dict

def train_test_models():
    """ Entry-point script to train and test models and save results to CSV.
    """
    output_pth = cfg.FOLDER_PATHS_DICTS["CROSS_VALIDATION"]
    
    def print_save_msg(path, file_type: str="file"):
        print(f"Sucessfully saved {file_type} to: {path}")
    asks_short_path = output_pth / "asks_ml_cross_val_30_min.csv"
    bids_short_path = output_pth / "bids_ml_cross_val_30_min.csv"
    
    start = time.time()
    
    #Read Data and Scale predictor variable
    scaler = MinMaxScaler((-100,100))
    asks_merged_df, bids_merged_df = fe.get_data(sin_cos_transform=True)
    for df in [asks_merged_df, bids_merged_df]:
        df[cfg.Y_COL] = scaler.fit_transform(df[cfg.Y_COL])
       
    # Get scores for short time horizon (frequency = 30 mins)
    print("============================================================")
    print("Running models for short time horizon (30 mins per interval)")
    asks_short_t_scores = get_cv_results(asks_merged_df, "30min")
    bids_short_t_scores = get_cv_results(bids_merged_df, "30min")
    print("Succesfully completed cross-validation for short time horizon models (30 mins per interval)")
    print("============================================================")
    print()
    
    # Get scores for long time horizon (frequency = daily)
    print("============================================================")
    print("Running models for short time horizon (daily interval)")
    scores_daily_models_dict = get_scores_daily_models(asks_merged_df, bids_merged_df)
    print("Succesfully completed cross-validation for long time horizon models (daily interval)")
    print("============================================================")
    print()
    
    print("============================================================")
    print("Saving files:")
    asks_short_t_scores.to_csv(asks_short_path)
    bids_short_t_scores.to_csv(bids_short_path)
    
    print_save_msg(asks_short_path, "CSV")
    print_save_msg(bids_short_path, "CSV")
    
    for key, val in scores_daily_models_dict.items():
        save_path = output_pth / f"{key}_ml_cross_val_1_day.csv"
        val.to_csv(save_path)
        print_save_msg(save_path, "CSV")
    end = round(time.time() / 60, 2)
    print(f"Succesfully trained all cross-validation in {end} minutes.")
    print("============================================================")
    return

     
if __name__ == "__main__":
    train_test_models()
    print("Succesfully executed ! Terminating program.")
