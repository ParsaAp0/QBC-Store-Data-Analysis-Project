from typing import Any
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from ..interface import model_interface
from .classification_grader import ClassificationGrader
from pathlib import Path
import json

target_feature = "Ship Mode"
input_columns = [
        "Segment",
        "Order Priority",
        "Order Date",
        "Region",
        "Market",
        "Total_Sales",
        "Total_Quantity",
        "Total_Profit",
        "Discount_Amount",
        "Sales_Per_Item",
        "Profit_Per_Item",
        "Category=Office Supplies",
        "Category=Technology",
        "Category=Furniture",
        "Category_mode",
        "Ship Mode"
]

all_params_path = Path("src/ML/models/classification/decision_tree_model_all_parameters.json")
best_params_path = Path("src/ML/models/classification/decision_tree_model_best_parameters.json")

class DecisionTreeModel(model_interface):
        def __init__(self):
                super().__init__("Decision Tree Classifier", "classification")

                self.target_column = target_feature
                self._load_best_params()

                # Model - Decision Tree
                self.model = DecisionTreeClassifier(
                        criterion=self.criterion,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=self.max_features,
                        random_state=100
                )
                self.label_encoder = LabelEncoder()
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                # self.scaler = RobustScaler()

                self.grader = ClassificationGrader()

                self.numeric_columns: list[str] = []
                self.categorical_columns: list[str] = []
                self.is_trained = False

        def train(self, train_data: pd.DataFrame) -> dict[str, Any]:
                X, y = self._prepare_data(train_data)
                X = self._feature_engineering(X)
                X = self._encode_features(X, fit=True)
                # X = self.scaler.fit_transform(X)
                y = self.label_encoder.fit_transform(y)
  
                X_train, X_val, y_train, y_val = train_test_split(
                        X, y, 
                        test_size=0.2, 
                        random_state=42, 
                        shuffle=False
                )
  
                class_weights = compute_class_weight(
                class_weight='balanced',
                        classes=np.unique(y_train),
                        y=y_train
                )
                weight_dict = dict(zip(np.unique(y_train), class_weights))
                sample_weights = np.array([weight_dict[label] for label in y_train])

                self.model.fit(X_train, y_train, sample_weight=sample_weights)
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
                # X = self.scaler.transform(X)

                predictions = self.model.predict(X)
                return y.to_numpy(), self.label_encoder.inverse_transform(predictions)

        def evaluate(self, test_data: pd.DataFrame) -> dict[str, float]:
                y, predictions = self.predict(test_data)
                score = self.grader.score(y, predictions)
                return score

        def tune(self, train_data: pd.DataFrame) -> dict:
                with all_params_path.open("r") as file:
                        parameter_grid = json.load(file)

                        param_grid = {
                                'criterion': parameter_grid.get('criterion', ['gini', 'entropy']),
                                'max_depth': parameter_grid.get('max_depth', [None, 5, 10, 15]),
                                'min_samples_split': parameter_grid.get('min_samples_split', [2, 5, 10]),
                                'min_samples_leaf': parameter_grid.get('min_samples_leaf', [1, 2, 4]),
                                'max_features': parameter_grid.get('max_features', [None, 'sqrt', 'log2']),
                        }
        
                        # Prepare original data
                        X, y = self._prepare_data(train_data)
                        
                        # Feature engineering
                        X = self._feature_engineering(X)
                
                        numeric_cols = X.select_dtypes(include=["number"]).columns
                        other_cols = [x for x in X.columns if x not in numeric_cols]
                
                        # Scaling (optional for decision tree, but keeping for consistency)
                        X = self._scaling(X, numeric_cols, other_cols)
                
                        # Encode categorical features
                        X = self._encode_features(X, fit=True)

                        X_train, X_val, y_train, y_val = train_test_split(
                                X, y, 
                            test_size=0.2, 
                               random_state=42, 
                          shuffle=False
                           )
   
                        grid = GridSearchCV(
                                estimator=DecisionTreeClassifier(random_state=42),
                                param_grid=param_grid,
                                scoring='f1_macro',
                                cv=5,
                                n_jobs=-1,
                                verbose=10,
                                error_score=0.0
                        )
                        grid.fit(
                                X_train, self.label_encoder.fit_transform(y_train)
                        )

                        print(
                                f"Best params: {grid.best_params_}, "
                                f"CV f1_macro: {grid.best_score_:.4f}"
                        )

                        best_score = grid.best_score_
                        best_parameters = grid.best_params_

                        with best_params_path.open("w") as file:
                                json.dump(best_parameters, file, indent=4)

                        print("------------ tuning result ------------")
                        print(f"Best parameters: {best_parameters}")
                        print(f"Best CV f1_macro: {best_score:.4f}")

                return best_parameters

        def get_parameters(self) -> dict:
                return {
                        "model": self.name,
                        "target_column": self.target_column,
                        "criterion": self.criterion,
                        "max_depth": self.max_depth,
                        "min_samples_split": self.min_samples_split,
                        "min_samples_leaf": self.min_samples_leaf,
                        "max_features": self.max_features,
                }

        def _prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
                if self.target_column not in data.columns:
                        raise ValueError(
                                f"Target column '{self.target_column}' does not exist.")

                data = data.copy()[input_columns]
                X = data.drop(columns=[self.target_column]).copy()
                y = data[self.target_column].copy()

                return X, y

        def _feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
                data = data.copy()
  
                  # Process date columns
                data['Order Date'] = pd.to_datetime(data['Order Date'])
                data['Order Day Of Month'] = data['Order Date'].dt.day
                data['Order Day Of Week'] = data['Order Date'].dt.day_of_week
                data['Order Month'] = data['Order Date'].dt.month
                data['Order Year'] = data['Order Date'].dt.year
  
                # Droping unwanted columns
                data.drop(columns=['Order Date'], inplace=True)
  
                return data

        def _scaling(self, data: pd.DataFrame, numeric_cols, other_cols) -> pd.DataFrame:        
                scaler = RobustScaler()
                data_scaled = pd.DataFrame(
                        scaler.fit_transform(data[numeric_cols]), 
                        columns = numeric_cols,
                        index = data.index
                )
        
                return pd.concat([data_scaled, data[other_cols]], axis=1)

        def _encode_features(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
                data = data.copy()

                if fit:
                        self.numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
                        self.categorical_columns = [
                                x for x in data.select_dtypes(include=["object", "string", "str", "category"]).columns.tolist() 
                                        if x not in ['Order Priority']
                        ]

                numeric_data = data[self.numeric_columns].copy()
                categorical_data = pd.DataFrame(index=data.index)

                if self.categorical_columns:
                        if fit:
                                encoded = self.encoder.fit_transform(data[self.categorical_columns])
                        else:
                                encoded = self.encoder.transform(data[self.categorical_columns])

                        encoded_columns = self.encoder.get_feature_names_out(
                                self.categorical_columns)

                        categorical_data = pd.DataFrame(
                                encoded,
                                columns=encoded_columns,
                                index=data.index,
                        )
        
                        categorical_data['Order Priority'] = data['Order Priority']\
                                .map(lambda x: 1 if x == 'Low' else 2 if x == 'Medium' else 3 if x == 'High' else 4)

                return pd.concat([numeric_data, categorical_data], axis=1)

        def _load_best_params(self):
                self.criterion = 'gini'
                self.max_depth = None
                self.min_samples_split = 2
                self.min_samples_leaf = 1
                self.max_features = None

                if not best_params_path.exists():
                        return

                with best_params_path.open("r") as file:
                        params = json.load(file)
                        self.criterion = params.get("criterion", self.criterion)
                        self.max_depth = params.get("max_depth", self.max_depth)
                        self.min_samples_split = params.get("min_samples_split", self.min_samples_split)
                        self.min_samples_leaf = params.get("min_samples_leaf", self.min_samples_leaf)
                        self.max_features = params.get("max_features", self.max_features)