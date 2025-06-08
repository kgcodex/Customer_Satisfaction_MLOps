import logging
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

class Evaluation(ABC):
    """
    Abstract class for evaluation metrics.
    """

    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate evaluation scores.
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        
        """
        pass


class MSE(Evaluation):
    """
    Mean Squared Error evaluation metric.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Mean Squared Error.
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        
        
        """
        
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating Mean Squared Error: {e}")
            raise e
        
class R2Score(Evaluation):
    """
    R-squared evaluation metric.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate R-squared score.
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        
        
        """
        
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R-squared Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R-squared Score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Root Mean Squared Error evaluation metric.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate Root Mean Squared Error.
        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
        
        
        """
        
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating Root Mean Squared Error: {e}")
            raise e
        
