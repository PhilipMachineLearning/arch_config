""" Machine Learning Utils for modelling and prediction.

Source: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from config import RANDOM_STATE, N_ESTIMATORS


MODELS = {
    # Train in order of computational complexity
    #"RandomForestRegressor": RandomForestRegressor(),
    #"AdaBoostRegressor": AdaBoostRegressor(),
    #"BaggingRegressor": BaggingRegressor(),
    "ExtraTreesRegressor": ExtraTreesRegressor(),
    #"SVR": SVR(), #SVR too slow on local machine high space/time complexity
    #"catboostregressor": CatBoostRegressor(),
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

    def fit(self, X, y, cv=10, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(
                model,
                params,
                cv=cv,
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
            print(k)
            params = self.grid_searches[k].cv_results_["params"]
            scores = []
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
