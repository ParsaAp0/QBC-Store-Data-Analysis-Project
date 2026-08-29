from data.DataLoader import DataLoader
from models.trainer import Trainer
from models.regression.example_model import SimpleRegressionModel
from models.comparator import ModelComparator
from models.selector import ModelSelector
import json

if __name__ == "__main__":
        loader = DataLoader(
                excel_location="data/data.xlsx",
                classification_excel_location = "data/classification_data.xlsx",
                train_test_split_percentage=0.8
        )

        loader.load()
        valid, log = loader.validate()
        print(log)
        if not valid:
                raise RuntimeError("Dataset validation failed.")
        
        
        regression_train_df = loader.get_regression_train_dataframe()
        regression_test_df = loader.get_regression_test_dataframe()
        classification_train_df = loader.get_classificaation_train_dataframe()
        classification_test_df = loader.get_classification_test_dataframe()
        
        trainer = Trainer(
                regression_models=[
                        SimpleRegressionModel(),
                        # XGBoostClassifier(),
                        # RandomForestRegressor(),
                ],
                classification_models=[
                        
                ]
        )

        results = trainer.train(
                # train_data=train_df,
                # test_data=test_df,
                regression_train_data = regression_train_df,
                regression_test_data = regression_test_df,
                classification_train_data = classification_train_df,
                classification_test_data = classification_test_df
        )


        
        
        metrics = {
                "classification": "f1",
                "regression": "rmse",
        }
        
        comparator = ModelComparator()

        selector = ModelSelector(
                comparator=comparator,
                experiment_name="Retail Project",
        )
        
        best_classifier = selector.select(
                task="classification",
                metric="f1",
        )

        best_regressor = selector.select(
                task="regression",
                metric="rmse",
        )
        
        print("-------------------------- best_classifier --------------------------")
        print(json.dumps(best_classifier, indent=4))
        print("-------------------------- best_regressor --------------------------")
        print(json.dumps(best_regressor, indent=4))

# uv sync
# uv run mlflow server --port 5000
# uv run src/ML/train.py 
