from data.DataLoader import DataLoader
from models.trainer import Trainer
from models.interface import grader_interface, model_interface
from models.classification.classification_grader import ClassificationGrader
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