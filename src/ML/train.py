from data.DataLoader import DataLoader


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
        print(f"{len(train_df) = }")
        print(f"{len(test_df) = }")