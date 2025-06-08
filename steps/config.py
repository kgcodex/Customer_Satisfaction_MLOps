from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    Configuration for the model name.

    """
    model_name: str = "LinearRegression"