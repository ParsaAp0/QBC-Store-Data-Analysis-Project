from data.DataLoader import DataLoader
from models.trainer import Trainer
from models.regression.example_model import SimpleRegressionModel


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
        
        # uv sync
        # uv run src/ML/train.py 
        # uv run mlflow server --port 5000
