from data.DataLoader import DataLoader
from models.trainer import Trainer
from models.regression.example_model import SimpleRegressionModel
from models.comparator import ModelComparator
from models.selector import ModelSelector
import json

if __name__ == "__main__":
        loader = DataLoader(
                excel_location="data/data.xlsx",
                train_test_split_percentage=0.8
        )

        loader.load()
        valid, log = loader.validate()
        print(log)
        if not valid:
                raise RuntimeError("Dataset validation failed.")

        train_df = loader.get_train_dataframe()
        test_df = loader.get_test_dataframe()
        
        trainer = Trainer(
                models=[
                        SimpleRegressionModel(),
                        # XGBoostClassifier(),
                        # RandomForestRegressor(),
                ]
        )

        results = trainer.train(
                train_data=train_df,
                test_data=test_df,
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
        # uv run src/ML/train.py 
        # uv run mlflow server --port 5000
