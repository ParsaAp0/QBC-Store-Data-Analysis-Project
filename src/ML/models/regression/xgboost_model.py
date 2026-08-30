from typing import Any
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
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

all_params_path = Path("src/ML/models/regression/xgboost_model_all_parameters.json")
best_params_path = Path("src/ML/models/regression/xgboost_model_best_parameters.json")

class XGBoostModel(model_interface):
        def __init__(self):
                super().__init__("XGBoost Regression", "regression")

                # Default hyperparameters
                self.target_column = target_feature
                self._load_best_params()
                
                # Model
                self.model = XGBRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        subsample=self.subsample,
                        colsample_bytree=self.colsample_bytree,
                        reg_alpha = self.reg_alpha,
                        reg_lambda = self.reg_lambda,
                        early_stopping_rounds=50,
                        random_state=100,
                        verbosity=0
                )

                # Preprocessing
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

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

                # 5. Training
                X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=42, shuffle=False  # Keep chronological order!
                )
                
                self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                )

                self.is_trained = True

                # Training score
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
                        'n_estimators': parameter_grid.get('n_estimators', [100]),
                        'max_depth': parameter_grid.get('max_depth', [6]),
                        'learning_rate': parameter_grid.get('learning_rate', [0.1]),
                        'subsample': parameter_grid.get('subsample', [0.7]),
                        'colsample_bytree': parameter_grid.get('colsample_bytree', [0.7])
                }

                # Prepare original data
                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)
                
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

                grid = GridSearchCV(
                        estimator=XGBRegressor(
                                random_state=42,
                                verbosity=0,
                                tree_method='hist',
                                device='cuda',
                                early_stopping_rounds=50
                        ),
                        param_grid=param_grid,
                        scoring='neg_root_mean_squared_error',
                        cv=3,
                        n_jobs=3,
                        verbose=10
                )
                grid.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                )
                
                best_score = -grid.best_score_
                best_parameters = grid.best_params_
                
                with best_params_path.open("w") as file:
                        json.dump(
                                best_parameters,
                                file,
                                indent=4,
                        )

                print("------------ tuning result ------------")
                print(f"Best parameters: {best_parameters}")
                print(f"Best Score: {best_score}")

                return best_parameters
        
        def get_parameters(self) -> dict:
                return {
                        "model": self.name,
                        "target_column": self.target_column,
                        "n_estimators": self.n_estimators,
                        "max_depth": self.max_depth,
                        "learning_rate": self.learning_rate,
                        "subsample": self.subsample,
                        "colsample_bytree": self.colsample_bytree,
                        "reg_alpha": self.reg_alpha,
                        "reg_lambda": self.reg_lambda,
                }

        def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                if self.target_column not in data.columns:
                        raise ValueError(f"Target column '{self.target_column}' does not exist.")

                data = data.copy()[input_columns]
                X = data.drop(columns=[self.target_column]).copy()
                y = data[self.target_column].copy()
                return X, y

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

                return pd.concat([numeric_data,categorical_data], axis=1)
        
        def _load_best_params(self):
                self.n_estimators = 100
                self.max_depth = 6
                self.learning_rate = 0.1
                self.subsample = 1.0
                self.colsample_bytree = 1.0
                self.reg_alpha = 1
                self.reg_lambda = 5
                
                if not best_params_path.exists():
                        return
                
                with best_params_path.open("r") as file:
                        params = json.load(file)
                self.n_estimators = params.get("n_estimators", self.n_estimators)
                self.max_depth = params.get("max_depth", self.max_depth)
                self.learning_rate = params.get("learning_rate", self.learning_rate)
                self.subsample = params.get("subsample", self.subsample)
                self.colsample_bytree = params.get("colsample_bytree", self.colsample_bytree)
                self.reg_alpha = params.get("reg_alpha", self.reg_alpha)
                self.reg_lambda = params.get("reg_lambda", self.reg_lambda)