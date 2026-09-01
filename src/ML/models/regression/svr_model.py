from typing import Any
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ..interface import model_interface
from sklearn.model_selection import GridSearchCV
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
	"Ship Date",
    "Profit",
]

all_params_path = Path("src/ML/models/regression/svr_all_parameters.json")
best_params_path = Path("src/ML/models/regression/svr_best_parameters.json")

class SVRModel(model_interface):
    def __init__(self):
        super().__init__("Support Vector Regressor", "regression")

        # Default hyperparameters
        self.target_column = target_feature
        self._load_best_params()

        # Model
        self.model = SVR(
            kernel=self.kernel,
            degree=self.degree,
            C=self.C,
            epsilon=self.epsilon
        )

        # Preprocessing
        self.encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False)
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

        # 3. Scaling
        X = self.scaler.fit_transform(X)

        # 4. Training
        self.model.fit(X, y)

        self.is_trained = True

        # Training score
        predictions = self.model.predict(X)
        score = self.grader.score(y.to_numpy(), predictions)

        return score

    def predict(self, test_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        X, y = self._prepare_data(test_data)

        # Use exactly the same feature engineering.
        X = self._feature_engineering(X)

        # Use already-fitted transformations.
        X = self._encode_features(X, fit=False)

        X = self.scaler.transform(X)

        predictions = self.model.predict(X)
        return y.to_numpy(), predictions

    def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
        y, predictions = self.predict(test_data)
        score = self.grader.score(y, predictions)

        return score

    def tune(self, train_data: pd.DataFrame) -> dict:
        with all_params_path.open("r") as file:
            parameters = json.load(file)

            kernels = parameters

            # Prepare original data
            X, y = self._prepare_data(train_data)

            # Feature engineering
            X = self._feature_engineering(X)

            # Encode categorical features
            X = self._encode_features(X, fit=True)

            # Search
            best_score = float("inf")
            best_parameters = {}

            for kernel in kernels:
                # Scaling
                scaler = StandardScaler()
                X_scale = scaler.fit_transform(X)

                # GridSearchCV
                grid = GridSearchCV(
                    estimator=SVR(),
                    param_grid=kernel,
                    scoring="neg_root_mean_squared_error",
                    cv=5,
                    n_jobs=-1,
                    verbose=10
                )

                grid.fit(X_scale, y)
                current_score = -grid.best_score_

                print(
                    f"Best kernel: {grid.best_params_['kernel']}, "
                    f"RMSE: {current_score}"
                )

                # Keep global best
                if current_score < best_score:
                    best_score = current_score
                    best_parameters = {
                        "kernel": grid.best_params_["kernel"],
                        "C": grid.best_params_["C"]
                    }
                    
                    if kernel['kernel'] == 'poly':
                        best_parameters.update({
                            "degree": grid.best_params_["degree"],
                            "coef0": grid.best_params_["coef0"]
                        })
                    else:
                        best_parameters.update({
                            "epsilon": grid.best_params_["epsilon"]
                        })

            # Save best parameters
            with best_params_path.open("w") as file:
                json.dump(
                    best_parameters,
                    file,
                    indent=4,
                )

            print("------------ tuning result ------------")
            print(f"Best parameters: {best_parameters}")
            print(f"Best CV RMSE: {best_score}")

            return best_parameters

    def get_parameters(self) -> dict:
        return {
            "model": self.name,
            "target_column": self.target_column,
            "kernel": self.kernel,
            "degree": self.degree,
            "C": self.C,
            "epsilon": self.epsilon
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
        
        # Cleaning outliers
        columns_to_check = ['Sales', 'Shipping Cost']
        mask = pd.DataFrame(index=X.index)

        for col in columns_to_check:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask[col] = (X[col] < lower_bound) | (X[col] > upper_bound)

        rows_to_drop = mask.any(axis=1)
        X = X[~rows_to_drop]
        y = y[~rows_to_drop]

        return X, y

    def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        # Example feature engineering.
        #
        # Add domain-specific features here.
        # The function must return a DataFrame.

        # Calculate Total Sales
        data["Total_Sales"] = data["Sales"] * data["Quantity"]
        
        # Calculate Total Discounted Sales
        data["Total_Discounted_Sales"] = data["Total_Sales"] * (1 - data["Discount"])
        
        # Process Date Columns
        data['Order Date'] = pd.to_datetime(data['Order Date'])
        data['Ship Date'] = pd.to_datetime(data['Ship Date'])
        data['Order Processing Days'] = (data['Ship Date'] - data['Order Date']).dt.days
        data['Order Day Of Month'] = data['Order Date'].dt.day
        data['Order Day Of Week'] = data['Order Date'].dt.day_of_week
        data['Order Month'] = data['Order Date'].dt.month
        
        # Droping unwanted columns
        data.drop(columns=['Order Date', 'Ship Date'])

        return data

    def _encode_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        data = data.copy()

        if fit:
            self.numeric_columns = data.select_dtypes(
                include=["int64", "float64"]).columns.tolist()

            self.categorical_columns = (
                data.select_dtypes(
                    include=["object", "string", "str", "category", "bool"]).columns.tolist()
            )

        numeric_data = data[self.numeric_columns].copy()
        categorical_data = pd.DataFrame(index=data.index)

        if self.categorical_columns:
            if fit:
                encoded = self.encoder.fit_transform(
                    data[self.categorical_columns])
            else:
                encoded = self.encoder.transform(
                    data[self.categorical_columns])

            encoded_columns = self.encoder.get_feature_names_out(
                self.categorical_columns)

            categorical_data = pd.DataFrame(
                encoded,
                columns=encoded_columns,
                index=data.index,
            )

        return pd.concat([numeric_data, categorical_data], axis=1)

    def _load_best_params(self):
        # Default values:
        kernel = 'rbf'
        degree = 3
        C = 1
        coef0 = 1.0
        epsilon = 1

        if not best_params_path.exists():
            self.kernel = kernel
            self.degree = degree
            self.C = C
            self.coef0 = coef0
            self.epsilon = epsilon

        with best_params_path.open("r") as file:
            parameters = json.load(file)
            self.kernel = parameters.get("max_depth", kernel)
            self.degree = parameters.get("learning_rate", degree)
            self.C = parameters.get("reg_lambda", C)
            self.epsilon = parameters.get("n_estimators", epsilon)
