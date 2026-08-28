import tempfile
from pathlib import Path
import pandas as pd
from .interface import model_interface
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ML Project")

class Trainer:
        def __init__(self, models: list[model_interface]):
                self.models = models
                self.results = {}

        def train(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                for model in self.models:
                        print(f"Training model: {model.name}")
                        
                        with mlflow.start_run(run_name=model.name):
                                mlflow.log_params(model.get_parameters())
                                train_score = model.train(train_data)
                                mlflow.log_metrics({
                                        f"train_{key}": value
                                        for key, value in train_score.items()
                                })
                                test_score = model.evaluate(test_data)
                                mlflow.log_metrics({
                                        f"test_{key}": value
                                        for key, value in test_score.items()
                                })

                                self._save_model(model)
                                self.results[model.name] = {"train": train_score, "test": test_score}
                        print("------------ result ------------")
                        print(train_score)
                        print(test_score)

                return self.results
        
        def _save_model(self, model: model_interface) -> str:
                with tempfile.TemporaryDirectory() as temp_dir:
                        model_path = Path(temp_dir) / "model.joblib"
                        model.save(str(model_path))
                        mlflow.log_artifact(str(model_path), artifact_path="model")