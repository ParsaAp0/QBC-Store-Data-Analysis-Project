from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from ..interface import model_interface
from .regression_grader import RegressionGrader

target_feature = "Profit"

input_columns = [
        "Order Priority",
        "Sales",
        "Quantity",
        "Discount",
        "Market",
        "Profit",
]

class SimpleRegressionModel(model_interface):
        def __init__(self):
                super().__init__("SimpleRegression example", "regression")

                # Default hyperparameters
                self.target_column = target_feature
                self.ridge_alpha = 1.0
                self.polynomial_degree = 3

                # Model
                self.model = Ridge(alpha=self.ridge_alpha)

                # Preprocessing
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.polynomial_features = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
                self.scaler = StandardScaler()

                # Grader
                self.grader = RegressionGrader()

                # Training state
                self.numeric_columns: list[str] = []
                self.categorical_columns: list[str] = []
                self.is_trained = False

        def train(self, train_data: pd.DataFrame) -> dict[str, Any]:
                X, y = self._prepare_data(train_data)

                # 1. Feature Engineering
                X = self._feature_engineering(X)

                # 2. Encode categorical features
                X = self._encode_features(X, fit=True)

                # 3. Polynomial features
                X = self._polynomial_features(X, fit=True)

                # 4. Scaling
                X = self.scaler.fit_transform(X)

                # 5. Training
                self.model.fit(X, y)

                self.is_trained = True

                # Training score
                predictions = self.model.predict(X)
                score = self.grader.score(y, predictions)

                return score

        def predict(self, test_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                if not self.is_trained:
                        raise RuntimeError("Model has not been trained yet.")

                X, y = self._prepare_data(test_data)

                # Use exactly the same feature engineering.
                X = self._feature_engineering(X)

                # Use already-fitted transformations.
                X = self._encode_features(X, fit=False)

                X = self._polynomial_features(X, fit=False)

                X = self.scaler.transform(X)

                predictions = self.model.predict(X)
                return y, predictions
                
        def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
                y, predictions = self.predict(test_data)
                score = self.grader.score(y, predictions)

                return score

        def tune(self, train_data: pd.DataFrame) -> dict:

                return {}
        
        def get_parameters(self) -> dict:
                return {
                        "model": self.name,
                        "target_column": self.target_column,
                        "ridge_alpha": self.ridge_alpha,
                        "polynomial_degree": self.polynomial_degree,
                }

        def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                if self.target_column not in data.columns:
                        raise ValueError(
                                f"Target column '{self.target_column}' "
                                "does not exist."
                        )

                data = data.copy()[input_columns]

                X = data.drop(columns=[self.target_column]).copy()

                y = data[self.target_column].copy()

                return X, y

        def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
                data = data.copy()
                
                # Example feature engineering.
                #
                # Add domain-specific features here.
                # The function must return a DataFrame.

                data["Sales_per_Quantity"] = (data["Sales"] / data["Quantity"].replace(0, np.nan))
                data["Discounted_Sales"] = (data["Sales"] * (1 - data["Discount"]))
                return data

        def _encode_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
                data = data.copy()

                if fit:
                        self.numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

                        self.categorical_columns = (
                                data.select_dtypes(include=["object", "string", "str", "category", "bool"]).columns.tolist()
                        )

                numeric_data = data[self.numeric_columns].copy()
                categorical_data = pd.DataFrame(index=data.index)

                if self.categorical_columns:
                        if fit:
                                encoded = self.encoder.fit_transform(data[self.categorical_columns])
                        else:
                                encoded = self.encoder.transform(data[self.categorical_columns])

                        encoded_columns = self.encoder.get_feature_names_out(self.categorical_columns)

                        categorical_data = pd.DataFrame(
                                encoded,
                                columns=encoded_columns,
                                index=data.index,
                        )

                return pd.concat([numeric_data,categorical_data], axis=1)

        def _polynomial_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
                data = data.copy()
                numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
                other_columns = [column for column in data.columns if column not in numeric_columns]
                numeric_data = data[numeric_columns]

                if fit:
                        polynomial_data = self.polynomial_features.fit_transform(numeric_data)
                else:
                        polynomial_data = self.polynomial_features.transform(numeric_data)

                polynomial_columns = self.polynomial_features.get_feature_names_out(numeric_columns)
                polynomial_dataframe = pd.DataFrame(
                        polynomial_data,
                        columns=polynomial_columns,
                        index=data.index,
                )

                if other_columns:
                        return pd.concat([polynomial_dataframe, data[other_columns]], axis=1)

                return polynomial_dataframe