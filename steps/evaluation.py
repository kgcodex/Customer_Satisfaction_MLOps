import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from sklearn.base import RegressorMixin

from src.evaluation import MSE, RMSE, R2Score


@step
def evaluate_model(
    model:RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Evaluate the trained model using various metrics.
    Args:
        model (RegressorMixin): The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target variable.
    """

    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Initialize evaluation metrics
        mse = MSE()
        rmse = RMSE()
        r2_score = R2Score()

        # Calculate scores
        mse_score = mse.calculate_scores(y_test, y_pred)
        rmse_score = rmse.calculate_scores(y_test, y_pred)
        r2_score_value = r2_score.calculate_scores(y_test, y_pred)

        # Log the scores
        logging.info(f"Mean Squared Error: {mse_score}")
        logging.info(f"Root Mean Squared Error: {rmse_score}")
        logging.info(f"R-squared Score: {r2_score_value}")

        return r2_score_value,rmse_score

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e

