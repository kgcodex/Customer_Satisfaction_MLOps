import logging
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig) -> RegressorMixin:

    """
    Train a Linear Regression model.
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Test target variable.
    Returns:
        RegressorMixin: Trained Linear Regression model.
    """
    try:
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel().train(X_train, y_train)
            return model
        else:
            logging.error(f"Model {config.model_name} is not supported.")
            raise ValueError(f"Model {config.model_name} is not supported.")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e




    
