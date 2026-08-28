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

        def train(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                
                regression_train_data = self._regression_prepare_data(train_data)
                regression_test_data = self._regression_prepare_data(test_data)
                classification_train_data = self._classification_prepare_data(train_data)
                classification_test_data = self._classification_prepare_data(test_data)
                
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
                
        def _regression_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                return data
        
        def _classification_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                # print("------------------ TEST -----------------")
                # group_col = "Order ID"
                # dfg = data.groupby(group_col)
                # vals = [
                #         "APAC",
                #         "US",
                #         "LATAM",
                #         "EU",
                #         "EMEA",
                #         "Africa",
                #         "Canada"
                # ]
                # print(self._str_get_mode(dfg, "Market"))
                
                # result = pd.concat([mode_series, entropy_series, counts_df], axis=1)
                return data
        
        
        def _aggregate_numbers(self, dfg, num_col: str):
                """
                Groups by 'group_col' and returns a DataFrame with:
                        sum, mean, std, median for 'num_col'
                """
                result = dfg[num_col].agg(
                        [f'sum({num_col})', f'mean({num_col})', f'std({num_col})', f'med({num_col})']
                ).rename(columns={
                        f'sum({num_col})': f'sum({num_col})',
                        f'mean({num_col})': f'mean({num_col})',
                        f'std({num_col})': f'std({num_col})',
                        f'med({num_col})': f'med({num_col})'
                })
                # Note: pandas .agg automatically names columns, but we keep it explicit.
                return result
        
        def _str_get_mode(self, dfg, str_col: str) -> pd.Series:
                mode_series = dfg[str_col].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                mode_series.name = f'mode({str_col})'
                return mode_series
        
        def _str_get_entropy(self, dfg, str_col: str) -> pd.Series:
                entropy_series = dfg[str_col].apply(self._normalized_entropy)
                entropy_series.name = f'Normalized Entropy({str_col})'
                return entropy_series
        
        def _str_get_repeat(self, dfg, str_col: str, categories: list[str]):
                #    unstack to get categories as columns, fill missing with 0
                counts_df = dfg[str_col].value_counts().unstack(fill_value=0)
                # Ensure all categories exist (reindex with the provided list)
                counts_df = counts_df.reindex(columns=categories, fill_value=0)
                # Rename columns to numberOfRep(cat)
                counts_df.columns = [f'numberOfRep({cat})' for cat in categories]
                return counts_df
        
        def _normalized_entropy(self, series: pd.Series):
                n = len(series)
                if n == 0:
                        return np.nan
                counts = series.value_counts()
                k = len(counts)                     # number of unique categories
                if k <= 1:
                        return 0.0
                probs = counts / n
                entropy = -sum(p * np.log(p) for p in probs)
                max_entropy = np.log(k)
                return entropy / max_entropy