import mlflow
from pathlib import Path
import json
from .comparator import ModelComparator

class ModelSelector:
        def __init__(self, comparator: ModelComparator, experiment_name: str):
                self.comparator = comparator
                self.experiment_name = experiment_name

        def select(self, task: str, metric: str) -> dict:
                runs = mlflow.search_runs(
                        experiment_names=[self.experiment_name],
                        filter_string = f"tags.task = '{task}' AND metrics.test_{metric} >= 0"
                )

                if len(runs) == 0:
                        return {}
                        raise ValueError(
                                f"No runs found for task='{task}' "
                                f"with metric='{metric}'."
                        )

                metric_values = {
                        row["run_id"]: row[f"metrics.test_{metric}"]
                        for _, row in runs.iterrows()
                }

                best_run_id = self.comparator.compare(
                        metric_values,
                        metric,
                )

                best_run = runs[runs["run_id"] == best_run_id].iloc[0]
                
                selected = {
                        "task": task,
                        "metric": metric,
                        "metric_value": best_run[f"metrics.test_{metric}"],
                        "model_name": best_run["tags.model_name"],
                        "model_version": best_run["tags.model_version"],
                        "run_id": best_run["run_id"],
                        "model_uri": best_run["tags.model_uri"],
                }
                
                self._save(selected=selected)

                return selected
                
        def _save(self, selected: dict) -> None:
                if len(selected) == 0:
                        return
                
                with Path(f"src/ML/models/{selected["task"]}/best.json").open("w") as file:
                        json.dump(
                                selected,
                                file,
                                indent=4
                        )