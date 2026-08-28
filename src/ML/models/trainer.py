from datetime import datetime
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from .interface import model_interface
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Retail Project")

class Trainer:
        def __init__(self, models: list[model_interface]):
                self.models = models
                self.results = {}

        def train(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                for model in self.models:
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