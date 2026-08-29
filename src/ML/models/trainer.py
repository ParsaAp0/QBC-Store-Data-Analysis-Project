from datetime import datetime
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import numpy as np

from .interface import model_interface
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Retail Project")

class Trainer:
        def __init__(self, regression_models: list[model_interface], classification_models: list[model_interface]):
                self.regression_models = regression_models
                self.classification_models = classification_models
                self.results = {}

        def train(self,
                        regression_train_data: pd.DataFrame,
                        regression_test_data: pd.DataFrame,
                        classification_train_data: pd.DataFrame,
                        classification_test_data: pd.DataFrame
                ) -> dict:
                # regression_train_data = regression_train_df,
                # regression_test_data = regression_test_df,
                # classification_train_data = classification_train_df,
                # classification_test_data = classification_test_df
                
                # regression_train_data = self._regression_prepare_data(train_data)
                # regression_test_data = self._regression_prepare_data(test_data)
                # classification_train_data = self._classification_prepare_data(train_data)
                # classification_test_data = self._classification_prepare_data(test_data)
                
                result_regression = self._task_train(self.regression_models, regression_train_data, regression_test_data)
                result_classification = self._task_train(self.classification_models, classification_train_data, classification_test_data)
                result = result_regression | result_classification
                return result
                
        def _task_train(self, models: list[model_interface], train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                for model in models:
                        model_version = datetime.now().strftime("%Y.%m.%d.%H%M%S")
                        with mlflow.start_run(run_name=model.name) as run:
                                # Metadata
                                mlflow.set_tags({
                                        "model_name": model.name,
                                        "model_version": model_version,
                                        "task": model.task,
                                })
                                mlflow.log_params(model.get_parameters())
                                # Train
                                train_score = model.train(train_data)
                                mlflow.log_metrics({
                                        f"train_{key}": value
                                        for key, value in train_score.items()
                                })

                                # Evaluate
                                test_score = model.evaluate(test_data)
                                mlflow.log_metrics({
                                        f"test_{key}": value
                                        for key, value in test_score.items()
                                })

                                # Save
                                model_uri = self._save_model(model)
                                mlflow.set_tag("model_uri", model_uri)

                                # Results
                                self.results[model.name] = {
                                        "task": model.task,
                                        "version": model_version,
                                        "run_id": run.info.run_id,
                                        "model_uri": model_uri,
                                        "train": train_score,
                                        "test": test_score,
                                }

                                print(f"------------ Training model: {model.name} ------------")
                                print(train_score)
                                print(test_score)
                                print(f"Version: {model_version}")
                                print(f"Run ID: {run.info.run_id}")
                                print(f"Model URI: {model_uri}")

                return self.results

        def _save_model(self, model: model_interface) -> str:
                with tempfile.TemporaryDirectory() as temp_dir:
                        model_path = Path(temp_dir) / "model.joblib"
                        model.save(str(model_path))
                        mlflow.log_artifact(
                                str(model_path),
                                artifact_path="model"
                        )
                        run_id = mlflow.active_run().info.run_id
                        return f"runs:/{run_id}/model/model.joblib"
                
        