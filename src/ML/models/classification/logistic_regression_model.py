from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.model_selection import GridSearchCV
from ..interface import model_interface
from .classification_grader import ClassificationGrader
from pathlib import Path
import json

target_feature = "Ship Mode"
input_columns = [
        "Segment",
        "Order Priority",
        "Region",
        "Market",
        "Sales_sum",
        "Sales_mean",
        "Sales_std",
        "Sales_median",
        "Quantity_sum",
        "Quantity_mean",
        "Quantity_std",
        "Quantity_median",
        "Discount_sum",
        "Discount_mean",
        "Discount_std",
        "Discount_median",
        "Category=Office Supplies",
        "Category=Technology",
        "Category=Furniture",
        "Category_entropy",
        "Category_mode",
        "Ship Mode"
]

all_params_path = Path("src/ML/models/classification/logistic_regression_model_all_parameters.json")
best_params_path = Path("src/ML/models/classification/logistic_regression_model_best_parameters.json")

class LogisticRegressionModel(model_interface):
        def __init__(self):
                super().__init__("Logistic Regression", "classification")

                self.target_column = target_feature
                self._load_best_params()
                
                # Model – using saga with l1_ratio (0=L2, 1=L1)
                self.model = LogisticRegression(
                        C=self.C,
                        solver='saga',
                        l1_ratio=self.l1_ratio,
                        max_iter=self.max_iter,
                        random_state=100
                )
                
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.polynomial_features = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False, interaction_only=True)
                self.scaler = StandardScaler()

                self.grader = ClassificationGrader()

                self.numeric_columns: list[str] = []
                self.categorical_columns: list[str] = []
                self.is_trained = False

        def train(self, train_data: pd.DataFrame) -> dict[str, Any]:
                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)
                X = self._polynomial_features(X, fit=True)
                X = self.scaler.fit_transform(X)

                self.model.fit(X, y)
                self.is_trained = True

                predictions = self.model.predict(X)
                score = self.grader.score(y, predictions)
                return score

        def predict(self, test_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                if not self.is_trained:
                        raise RuntimeError("Model has not been trained yet.")

                X, y = self._prepare_data(test_data)
                X = self._feature_engineering(X)
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
                with all_params_path.open("r") as file:
                        parameters = json.load(file)
                        
                C_values = parameters.get("C", [0.01, 0.1, 1, 10])
                polynomial_degrees = parameters.get("polynomial_degree", [1, 2])
                l1_ratios = parameters.get("l1_ratio", [0.0, 0.5, 1.0])
                max_iters = parameters.get("max_iter", [500, 1000, 2000])
                

                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)

                best_score = float("inf")
                best_parameters = {}

                for degree in polynomial_degrees:
                        poly = PolynomialFeatures(
                                degree=degree,
                                include_bias=False,
                                interaction_only=True
                        )
                        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
                        other_cols = [col for col in X.columns if col not in numeric_cols]
                        numeric_data = X[numeric_cols]

                        poly_data = poly.fit_transform(numeric_data)
                        poly_cols = poly.get_feature_names_out(numeric_cols)
                        poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=X.index)

                        if other_cols:
                                X_poly = pd.concat([poly_df, X[other_cols]], axis=1)
                        else:
                                X_poly = poly_df

                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_poly)

                        param_grid = {
                                'C': C_values,
                                'l1_ratio': l1_ratios,
                                'max_iter': max_iters
                        }

                        grid = GridSearchCV(
                                estimator=LogisticRegression(
                                        # C=self.C,
                                        solver='saga',
                                        # l1_ratio=self.l1_ratio,
                                        # max_iter=self.max_iter,
                                        random_state=100,
                                ),
                                param_grid=param_grid,
                                scoring='accuracy',
                                cv=5,
                                n_jobs=3,
                                verbose=10,
                                error_score=0.0
                        )

                        grid.fit(X_scaled, y)
                        current_score = grid.best_score_

                        print(
                                f"Degree: {degree}, "
                                f"Best params: {grid.best_params_}, "
                                f"CV accuracy: {current_score:.4f}"
                        )

                        if current_score > best_score:
                                best_score = current_score
                                best_parameters = {
                                        "C": grid.best_params_["C"],
                                        "l1_ratio": grid.best_params_["l1_ratio"],
                                        "polynomial_degree": degree
                                }

                with best_params_path.open("w") as file:
                        json.dump(best_parameters, file, indent=4)

                print("------------ tuning result ------------")
                print(f"Best parameters: {best_parameters}")
                print(f"Best CV accuracy: {best_score:.4f}")

                return best_parameters
        
        def get_parameters(self) -> dict:
                return {
                        "model": self.name,
                        "target_column": self.target_column,
                        "C": self.C,
                        "l1_ratio": self.l1_ratio,
                        "max_iter": self.max_iter,
                        "polynomial_degree": self.polynomial_degree,
                }

        def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                if self.target_column not in data.columns:
                        raise ValueError(f"Target column '{self.target_column}' does not exist.")
                
                data = data.copy()[input_columns]
                X = data.drop(columns=[self.target_column]).copy()
                y = data[self.target_column].copy()
                return X, y

        def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
                # No feature engineering for now – kept as placeholder
                return data.copy()

        def _encode_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
                data = data.copy()

                if fit:
                        self.numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
                        self.categorical_columns = data.select_dtypes(include=["object", "string", "str", "category", "bool"]).columns.tolist()

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

                return pd.concat([numeric_data, categorical_data], axis=1)

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
        
        def _load_best_params(self):
                self.C = 1.0
                self.l1_ratio = 0.0
                self.max_iter = 1000
                self.polynomial_degree = 1

                if not best_params_path.exists():
                        return

                with best_params_path.open("r") as file:
                        parameters = json.load(file)
                self.C = parameters.get("C", self.C)
                self.l1_ratio = parameters.get("l1_ratio", self.l1_ratio)
                self.max_iter = parameters.get("max_iter", self.max_iter)
                self.polynomial_degree = parameters.get("polynomial_degree", self.polynomial_degree)