import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models.
    """

    @abstractmethod
    def train(self,X_train,y_train):
        """
        Train the model.
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target variable.
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Train the Linear Regression model.
        Args:
            X_train (pd.DataFrame): The training features.
            y_train (pd.Series): The training target variable.
        """
        try:
            self.model = LinearRegression(**kwargs)
            self.model.fit(X_train, y_train)
            logging.info("Linear Regression model trained successfully.")
            return self.model
        
        except Exception as e:
            logging.error(f"Error in training Linear Regression model: {e}")
            raise e
