from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib

class model_interface(ABC):
        def __init__(self, name: str):
                self.name = name
                self.model = None

        @abstractmethod
        def train(self, train_data: pd.DataFrame) -> dict:
                """
                Train the model.

                Returns:
                        A dictionary containing information about
                        the training process and its results.
                """
                pass

        @abstractmethod
        def evaluate(self, test_data: pd.DataFrame) -> dict:
                """
                Evaluate the trained model on the test dataset.

                Returns:
                        A dictionary containing evaluation scores.
                """
                pass

        @abstractmethod
        def tune(self, train_data: pd.DataFrame) -> dict:
                """
                Tune the model's hyperparameters.

                Returns:
                        The best parameters found during tuning.
                """
                pass
        
        @abstractmethod
        def predict(self, test_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                pass
        
        def save(self, path: str) -> None:
                joblib.dump(self, path)

        @classmethod
        def load(cls, path: str):
                return joblib.load(path)
        
        @abstractmethod
        def get_parameters(self) -> dict:
                pass

class grader_interface(ABC):

        @abstractmethod
        def score(
                self,
                y_true: np.ndarray,
                y_predict: np.ndarray,
        ) -> dict:
                pass