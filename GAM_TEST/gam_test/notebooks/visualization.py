""" Visualization Utils for modelling and prediction.

Used to save plots for ML models.

Source: http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
"""
import pandas as pd
import numpy as np
import ml_modelling as ml
import matplotlib.pyplot as plt
import config as cfg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import mean_squared_error
import shap

STRING_IMAGE = "best_ml_model_interval_"
FEATURES_PDP = ('Volume', 'Time')

def train_models(
    models_dict:dict,
    X_train:pd.DataFrame, 
    y_train:pd.DataFrame,
    X_test:pd.DataFrame, 
    y_test:pd.DataFrame,
)-> dict:
    """ Utility function to train models for visualization
    :return: Updated dictionary with trained models
    """
    y_train, y_test= y_train.values.ravel(), y_test.values.ravel()
    for key, val in models_dict.items():  
        print(f"Fitting model: {key}")
        model = val["model"].fit(X_train, y_train)
        y_preds = model.predict(X_test)
        mse = round(mean_squared_error(y_test, y_preds), 2)
        val["model"] = model
        val["mse"] = mse
        print(f"MSE for {key}: {mse}")
    return models_dict

def get_partial_dependence_plots(model, X_input,save_path, features=FEATURES_PDP):
    def print_save_msg(path, file_type: str):
        print(f"Sucessfully saved {file_type} to: {path}")
        
    print("Generating plots")
    fig = plt.figure(figsize=(15,10))
    pdp, axes = partial_dependence(model, X_input, features=features,
                                   grid_resolution=20)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of scaled liquidity cost on\n'
                 'volume and time')
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=300)
    print_save_msg(save_path, "image")
    return ax, plt

def shap_summary_plots(model, X_input):
    """ Draw SHAP summary plot
    :model: ML model
    :X_input: Input
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    
    # summarize the effects of all the features
    shap.summary_plot(shap_values, X_input)
    return