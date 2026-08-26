import pandas as pd
from .interface import model_interface

class Trainer:
        def __init__(self, models: list[model_interface]):
                self.models = models
                self.results = {}

        def train(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> dict:
                for model in self.models:
                        print(f"Training model: {model.name}")

                        train_log = model.train(train_data)
                        test_score = model.evaluate(test_data)

                        self.results[model.name] = {
                                "train": train_log,
                                "test": test_score,
                        }

                return self.results