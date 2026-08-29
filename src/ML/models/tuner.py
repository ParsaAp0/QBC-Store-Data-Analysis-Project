from datetime import datetime
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np

from .interface import model_interface
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Retail Project")

class Tuner:
        def __init__(self, regression_models: list[model_interface], classification_models: list[model_interface]):
                self.regression_models = regression_models
                self.classification_models = classification_models
                self.results = {}

        def tune(self,
                        regression_train_data: pd.DataFrame,
                        regression_test_data: pd.DataFrame,
                        classification_train_data: pd.DataFrame,
                        classification_test_data: pd.DataFrame
                ) -> dict:
                
                result_regression = self._task_tune(self.regression_models, regression_train_data, regression_test_data)
                result_classification = self._task_tune(self.classification_models, classification_train_data, classification_test_data)
                result = result_regression | result_classification
                return result
                
        def _task_tune(self, models: list[model_interface], train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                for model in models:
                        tune_score = model.tune(train_data)

                return self.results
        