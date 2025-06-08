import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessingStrategy

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:

    """
    Clean the data and divide it into train and test sets.
    Args:
        df (pd.DataFrame): The input data to be cleaned and divided.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The train and test sets for features and target variable.
    """

    try:
        process_strategy= DataPreprocessingStrategy()
        data_cleaning= DataCleaning(df, process_strategy)
        processed_data=data_cleaning.handle_data()

        divide_strategy= DataDivideStrategy()
        data_cleaning= DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        return X_train, X_test, y_train, y_test
        logging.info("Data cleaning and division completed successfully.")

    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
