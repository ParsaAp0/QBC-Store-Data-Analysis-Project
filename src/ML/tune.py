from data.DataLoader import DataLoader
from models.tuner import Tuner
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
        
        tuner = Tuner(
                regression_models=[
                        SimpleRegressionModel(),
                        # XGBoostClassifier(),
                        # RandomForestRegressor(),
                ],
                classification_models=[
                        
                ]
        )

        results = tuner.tune(
                # train_data=train_df,
                # test_data=test_df,
                regression_train_data = regression_train_df,
                regression_test_data = regression_test_df,
                classification_train_data = classification_train_df,
                classification_test_data = classification_test_df
        )
        
        print("-------------------------- Result --------------------------")
        print(json.dumps(results, indent=4))
