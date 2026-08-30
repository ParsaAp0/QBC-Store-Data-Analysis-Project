from typing import Any
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor  # <-- new import
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from ..interface import model_interface
from .regression_grader import RegressionGrader
from pathlib import Path
import json

target_feature = "Profit"

input_columns = [
        "Sales",
        "Quantity", 
        "Discount", 
        "Shipping Cost",
        "Order Priority",
        "Market",
        "Region",
        "Segment",
        "Ship Mode",
        "Category",
        "Sub-Category",
        "Order Date",
        "Profit"
]

# Change the file names to avoid overwriting your XGBoost parameters
all_params_path = Path("src/ML/models/regression/ebm_model_all_parameters.json")
best_params_path = Path("src/ML/models/regression/ebm_model_best_parameters.json")

class EBMModel(model_interface):   # renamed class
        def __init__(self):
                super().__init__("Explainable Boosting Machine", "regression")

                self.target_column = target_feature
                self._load_best_params()
                
                # EBM model with correct parameter names
                self.model = ExplainableBoostingRegressor(
                        max_bins=self.max_bins,
                        interactions=self.interactions,          # correct parameter name
                        learning_rate=self.learning_rate,
                        random_state=42,
                        n_jobs=-1
                )

                # Preprocessing (same as before)
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

                self.grader = RegressionGrader()

                self.numeric_columns: list[str] = []
                self.categorical_columns: list[str] = []
                self.is_trained = False

        def train(self, train_data: pd.DataFrame) -> dict[str, Any]:
                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)

                # Train on the full training set (no validation split needed)
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

                predictions = self.model.predict(X)
                return y, predictions
                
        def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
                y, predictions = self.predict(test_data)
                score = self.grader.score(y, predictions)
                return score

        def tune(self, train_data: pd.DataFrame) -> dict:
                with all_params_path.open("r") as file:
                        parameter_grid = json.load(file)

                param_grid = {
                        'max_bins': parameter_grid.get('max_bins', [128, 256]),
                        'interactions': parameter_grid.get('interactions', [0, 3, 5]),
                        'learning_rate': parameter_grid.get('learning_rate', [0.01, 0.05, 0.1])
                }

                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)

                grid = GridSearchCV(
                        estimator=ExplainableBoostingRegressor(random_state=42, n_jobs=-1),
                        param_grid=param_grid,
                        scoring='neg_root_mean_squared_error',
                        cv=3,
                        n_jobs=3,
                        verbose=10
                )
                grid.fit(X, y)
                
                best_score = -grid.best_score_
                best_parameters = grid.best_params_
                
                with best_params_path.open("w") as file:
                        json.dump(best_parameters, file, indent=4)

                print("------------ tuning result ------------")
                print(f"Best parameters: {best_parameters}")
                print(f"Best CV RMSE: {best_score}")

                return best_parameters
        
        def get_parameters(self) -> dict:
                return {
                        "model": self.name,
                        "target_column": self.target_column,
                        "max_bins": self.max_bins,
                        "interactions": self.interactions,
                        "learning_rate": self.learning_rate,
                }

        def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                if self.target_column not in data.columns:
                        raise ValueError(f"Target column '{self.target_column}' does not exist.")

                data = data.copy()[input_columns]
                X = data.drop(columns=[self.target_column]).copy()
                y = data[self.target_column].copy()
                return X, y

        # def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        #         data = data.copy()
        #         data["Sales_per_Quantity"] = data["Sales"] / data["Quantity"].replace(0, np.nan)
        #         data["Discounted_Sales"] = data["Sales"] * (1 - data["Discount"])
        #         # EBM can handle NaN, but we fill to be safe with scaling (though EBM doesn't need scaling)
        #         data.fillna(0, inplace=True)
        #         return data
        def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
                data = data.copy()
                data["Sales_per_Quantity"] = data["Sales"] / data["Quantity"].replace(0, np.nan)
                data["Discounted_Sales"] = data["Sales"] * (1 - data["Discount"])
                data["Shipping_Cost_per_Unit"] = data["Shipping Cost"] / data["Quantity"].replace(0, np.nan)
                data["Order Date"] = pd.to_datetime(data["Order Date"])
                data["Quarter"] = data["Order Date"].dt.quarter
                
                data.drop(columns=["Order Date"], inplace=True)
                data.fillna(0, inplace=True)
                return data

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
        
        def _load_best_params(self):
                # Defaults for EBM
                self.max_bins = 256
                self.interactions = 3
                self.learning_rate = 0.05
                
                if not best_params_path.exists():
                        return
                
                with best_params_path.open("r") as file:
                        params = json.load(file)
                self.max_bins = params.get("max_bins", self.max_bins)
                self.interactions = params.get("interactions", self.interactions)
                self.learning_rate = params.get("learning_rate", self.learning_rate)