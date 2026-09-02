from pathlib import Path
import numpy as np
import pandas as pd
import json

PROJECT_DIR = Path(__file__).parent.parent.parent.parent
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / 'configs'

excel_sheets_config = CONFIG_DIR / 'excel_sheets.json'
merges_config = CONFIG_DIR / 'excel_merges.json'
validation_columns = CONFIG_DIR / 'excel_columns.json'
order_column_name = "Order ID"


class DataLoader:
        def __init__(
                        self,
                        excel_location: str | Path,
                        classification_excel_location: str | Path | None,
                        train_test_split_percentage: float,
                        random_state: int = 42,
        ):
         
                self.excel_location = Path(PROJECT_DIR / excel_location)
                if (classification_excel_location is None):
                        self.classification_excel_location = None
                else:
                        self.classification_excel_location = Path(PROJECT_DIR / classification_excel_location)
                self.train_test_split_percentage = train_test_split_percentage
                with open(excel_sheets_config, 'r') as file:
                        self.excel_sheets = json.load(file)
                with open(merges_config, 'r') as file:
                        self.merges = json.load(file)
                with open(validation_columns, 'r') as file:
                        self.validation_columns = json.load(file)
                self.main_sheet = self.excel_sheets[0]
                self.random_state = random_state

                self._sheets: dict[str, pd.DataFrame] = {}
                self._dataframe: pd.DataFrame | None = None
                self._train_dataframe: pd.DataFrame | None = None
                self._test_dataframe: pd.DataFrame | None = None

        def load(self) -> None:
                """
                Load the configured Excel sheets, merge them, and
                split the resulting dataframe into train and test sets.
                """

                self._sheets = pd.read_excel(
                        self.excel_location,
                        sheet_name=self.excel_sheets,
                )
                self.classification_data = None
                if self.classification_excel_location is not None:
                        try:
                                self.classification_data = pd.read_excel(
                                        self.classification_excel_location,
                                        sheet_name="data",
                                )
                        except:
                                self.classification_data = None
                self._merge()
                self._split()
                self._prepare_data()

        def validate(self) -> tuple[bool, str]:
                """
                Validate the loaded Excel sheets.

                Validation rules:
                                1. All expected columns must exist.
                                2. Column types must match the expected types.
                                If a type is incorrect, an attempt is made to convert it.
                                3. No null values are allowed.

                Returns:
                                tuple[bool, str]:
                                                A boolean indicating whether validation passed and
                                                a human-readable validation log.
                """

                errors: list[str] = []
                fixes: list[str] = []

                if not self._sheets:
                        errors.append("No data has been loaded. Call load() first.")
                        return False, self._build_log(errors, fixes)

                # ---------------------------------------------------------
                # Validate every configured sheet
                # ---------------------------------------------------------
                for sheet_name, expected_columns in self.validation_columns.items():
                        # -------------------------------------------------
                        # Sheet existence
                        # -------------------------------------------------
                        if sheet_name not in self._sheets:
                                errors.append(
                                        f"Sheet '{sheet_name}' does not exist "
                                        "in the loaded data."
                                )
                                continue

                        dataframe = self._sheets[sheet_name]

                        # -------------------------------------------------
                        # Validate columns
                        # -------------------------------------------------
                        for column_name, expected_type in expected_columns.items():
                                if column_name not in dataframe.columns:
                                        errors.append(
                                                f"Sheet '{sheet_name}': expected "
                                                f"column '{column_name}' is missing."
                                        )
                                        continue
                                actual_type = str(dataframe[column_name].dtype)

                                # -------------------------------------------------
                                # Type validation / conversion
                                # -------------------------------------------------
                                if actual_type != expected_type:
                                        try:
                                                if expected_type == "str":
                                                        dataframe[column_name] = (
                                                                dataframe[column_name]
                                                                .astype("string")
                                                        )

                                                elif expected_type == "int64":
                                                        dataframe[column_name] = (
                                                                pd.to_numeric(
                                                                        dataframe[
                                                                                column_name
                                                                        ],
                                                                        errors="raise",
                                                                ).astype("int64")
                                                        )

                                                elif expected_type == "float64":
                                                        dataframe[column_name] = (
                                                                pd.to_numeric(
                                                                        dataframe[
                                                                                column_name
                                                                        ],
                                                                        errors="raise",
                                                                ).astype("float64")
                                                        )

                                                elif expected_type == "bool":
                                                        # Handle common boolean
                                                        # representations explicitly.
                                                        value_map = {
                                                                True: True,
                                                                False: False,
                                                                "True": True,
                                                                "False": False,
                                                                "true": True,
                                                                "false": False,
                                                                "TRUE": True,
                                                                "FALSE": False,
                                                                1: True,
                                                                0: False,
                                                                "1": True,
                                                                "0": False,
                                                        }
                                                        original = dataframe[column_name]
                                                        converted = original.map(value_map)
                                                        if (converted.isna() & original.notna()).any():
                                                                invalid_values = original[converted.isna(
                                                                ) & original.notna()].unique().tolist()
                                                                raise ValueError(
                                                                        f"Invalid boolean values: {invalid_values}")

                                                        dataframe[column_name] = converted.astype(bool)
                                                else:
                                                        raise TypeError(
                                                                f"Unsupported expected type '{expected_type}'."
                                                        )

                                                # Verify that the conversion actually
                                                # produced the expected dtype.
                                                converted_type = str(dataframe[column_name].dtype)

                                                # pandas uses "string" rather than
                                                # "str" for StringDtype.
                                                type_matches = (
                                                        expected_type == "str"
                                                        and converted_type
                                                        in {"string", "object"}
                                                ) or (
                                                        converted_type == expected_type
                                                )

                                                if not type_matches:
                                                        raise TypeError(
                                                                f"Conversion resulted "
                                                                f"in dtype "
                                                                f"'{converted_type}'."
                                                        )

                                                fixes.append(
                                                        f"Sheet '{sheet_name}': column "
                                                        f"'{column_name}' converted "
                                                        f"from '{actual_type}' to "
                                                        f"'{expected_type}'."
                                                )

                                        except (ValueError, TypeError, OverflowError) as exc:
                                                errors.append(
                                                        f"Sheet '{sheet_name}': column "
                                                        f"'{column_name}' has type "
                                                        f"'{actual_type}', expected "
                                                        f"'{expected_type}', and could "
                                                        f"not be converted: {exc}"
                                                )

                                # -------------------------------------------------
                                # Null validation
                                # -------------------------------------------------
                                dataframe.dropna(inplace=True)
                                if dataframe[column_name].isna().any():
                                        null_count = int(dataframe[column_name].isna().sum())

                                        errors.append(
                                                f"Sheet '{sheet_name}': column "
                                                f"'{column_name}' contains "
                                                f"{null_count} null value(s)."
                                        )

                        # -------------------------------------------------
                        # Update the stored dataframe after possible conversions
                        # -------------------------------------------------
                        self._sheets[sheet_name] = dataframe

                # ---------------------------------------------------------
                # Validate unexpected columns
                # ---------------------------------------------------------
                # Not required by the three requested conditions, so these
                # are intentionally allowed. The validation only guarantees
                # that all expected columns exist and have the correct types.
                # ---------------------------------------------------------

                valid = len(errors) == 0
                lines = [
                        "--------------------- DataLoader Validation ---------------------",
                ]

                if fixes:
                        lines.append("FIXES:")
                        for fix in fixes:
                                lines.append(f"  [FIXED] {fix}")
                        lines.append("")

                if errors:
                        lines.append("ERRORS:")
                        for error in errors:
                                lines.append(f"  [ERROR] {error}")
                        lines.append("")

                if valid:
                        lines.append("Validation: PASSED")
                else:
                        lines.append("Validation: FAILED")

                return valid, "\n".join(lines)

        def get_regression_train_dataframe(self) -> pd.DataFrame:
                if self._regression_train_dataframe is None:
                        raise RuntimeError(
                                "Dataset has not been loaded and split yet. Call load() first."
                        )

                return self._regression_train_dataframe.copy()

        def get_classificaation_train_dataframe(self) -> pd.DataFrame:
                if self._classification_train_dataframe is None:
                        raise RuntimeError(
                                "Dataset has not been loaded and split yet. Call load() first."
                        )

                return self._classification_train_dataframe.copy()

        def get_regression_test_dataframe(self) -> pd.DataFrame:
                if self._regression_test_dataframe is None:
                        raise RuntimeError(
                                "Dataset has not been loaded and split yet. Call load() first."
                        )

                return self._regression_test_dataframe.copy()

        def get_classification_test_dataframe(self) -> pd.DataFrame:
                if self._classification_test_dataframe is None:
                        raise RuntimeError(
                                "Dataset has not been loaded and split yet. Call load() first."
                        )

                return self._classification_test_dataframe.copy()

        # =========================================================
        # Private methods
        # =========================================================

        def _merge(self) -> None:
                """
                Merge all configured dimension tables into self.main_sheet.
                """

                if not self._sheets:
                        raise RuntimeError(
                                "No sheets have been loaded. Call load() first."
                        )

                if self.main_sheet not in self._sheets:
                        raise KeyError(
                                "self.main_sheet must be present in excel_sheets."
                        )

                dataframe = self._sheets[self.main_sheet].copy()
                for merge in self.merges:
                        right_name = merge["right"]
                        if right_name not in self._sheets:
                                raise KeyError(
                                        f"Sheet '{right_name}' was not loaded."
                                )

                        dataframe[merge["left_on"]] = dataframe[merge["left_on"]
                                                                                                        ].str.lower().str.strip()
                        self._sheets[right_name][merge["right_on"]
                                                                         ] = self._sheets[right_name][merge["right_on"]].str.lower().str.strip()

                        dataframe = dataframe.merge(
                                self._sheets[right_name],
                                left_on=merge["left_on"],
                                right_on=merge["right_on"],
                                how=merge["how"],
                        )
                self._dataframe = dataframe.dropna()

        def _prepare_data(self) -> None:
                self._regression_train_dataframe = self._regression_prepare_data(
                        self._train_dataframe)
                self._regression_test_dataframe = self._regression_prepare_data(
                        self._test_dataframe)
                if (self.classification_data is None):
                        self._classification_train_dataframe = self._classification_prepare_data(
                                self._train_dataframe)
                        self._classification_test_dataframe = self._classification_prepare_data(
                                self._test_dataframe)
                else:
                        return

                pd.concat([
                        self._classification_train_dataframe,
                        self._classification_test_dataframe
                ], ignore_index=True).to_excel('data/classification_data.xlsx', sheet_name='data', index=False)

        def _split(self) -> None:
                """
                Split the merged dataframe into train and test sets.
                """

                if self._dataframe is None:
                        raise RuntimeError(
                                "No merged dataframe exists. Call load() first.")

                # 1. CRITICAL: Sort by Order Date (Oldest to Newest)
                self._dataframe = self._dataframe.sort_values(
                        by="Order Date").reset_index(drop=True)

                # 2. Find the cut-off point (80% of the chronological data)
                split_idx = int(len(self._dataframe) *
                                                self.train_test_split_percentage)

                # 3. Split by position (No randomness!)
                train_dataframe = self._dataframe.iloc[:split_idx]
                test_dataframe = self._dataframe.iloc[split_idx:]

                self._train_dataframe = train_dataframe.reset_index(drop=True)
                self._test_dataframe = test_dataframe.reset_index(drop=True)

                # If you have classification data, do the exact same thing:
                if self.classification_data is not None:
                        # Ensure classification data is also sorted consistently with the main df
                        # Since it shares the same index, just use the same split_idx
                        self.classification_data = self.classification_data.sort_values(
                                by="Order Date").reset_index(drop=True)
                        class_split_idx = int(
                                len(self.classification_data) * self.train_test_split_percentage)
                        train_dataframe = self.classification_data.iloc[:class_split_idx]
                        test_dataframe = self.classification_data.iloc[class_split_idx:]

                        self._classification_train_dataframe = train_dataframe.reset_index(
                                drop=True)
                        self._classification_test_dataframe = test_dataframe.reset_index(
                                drop=True)

        @staticmethod
        def _build_log(errors: list[str], warnings: list[str]) -> str:
                lines = [
                        "--------------------- DataLoader Validation ---------------------",
                ]

                if errors:
                        lines.append("ERRORS:")

                        for error in errors:
                                lines.append(f"  [ERROR] {error}")

                        lines.append("")

                if warnings:
                        lines.append("WARNINGS:")

                        for warning in warnings:
                                lines.append(f"  [WARNING] {warning}")
                        lines.append("")

                if errors:
                        lines.append("Validation: FAILED")
                else:
                        lines.append("Validation: PASSED")
                return "\n".join(lines)

        def _regression_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                return data

        def _classification_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
                group_col = "Order ID"
                df_grouped = data.groupby(group_col)

                # customer_id = self._str_get_mode(df_grouped, "Customer ID", "Customer ID")
                segment = self._str_get_mode(df_grouped, "Segment", "Segment")
                order_priority = self._str_get_mode(df_grouped, "Order Priority", "Order Priority")
                order_date = self._str_get_mode(df_grouped, "Order Date", "Order Date")
                # city = self._str_get_mode(df_grouped, "City", "City")
                # state = self._str_get_mode(df_grouped, "State", "State")
                # country = self._str_get_mode(df_grouped, "Country", "Country")
                region = self._str_get_mode(df_grouped, "Region", "Region")
                market = self._str_get_mode(df_grouped, "Market", "Market")
                sales_cols = self._aggregate_numbers(df_grouped, "Sales")
                quantity_cols = self._aggregate_numbers(df_grouped, "Quantity")
                discount_cols = self._aggregate_numbers(df_grouped, "Discount")
                profit_cols = self._aggregate_numbers(df_grouped, "Profit")
                numeric_cols = self._aggregate_numeric_cols(df_grouped)
                category_repeat = self._str_get_repeat(df_grouped, "Category", data["Category"].unique())
                category_entropy = self._str_get_entropy(df_grouped, "Category")
                category_mode = self._str_get_mode(df_grouped, "Category")
                ship_mode = self._str_get_mode(df_grouped, "Ship Mode", "Ship Mode")

                result = pd.concat([
                        # customer_id,
                        segment,
                        order_priority,
                        order_date,
                        # city,
                        # state,
                        # country,
                        region,
                        # market_repeat,
                        # market_entropy,
                        market,
                        sales_cols,
                        quantity_cols,
                        discount_cols,
                        profit_cols,
                           numeric_cols,
                        category_repeat,
                        category_entropy,
                        category_mode,
                        ship_mode
                ], axis=1)

                result.reset_index(inplace=True)
                return result

        def _aggregate_numbers(self, df_grouped, num_col: str):
                """
                Groups by 'group_col' and returns a DataFrame with:
                                sum, mean, std, median for 'num_col'
                """
                result = df_grouped[num_col].agg(
                        ['sum', 'mean', 'std', 'median']
                ).fillna(0).rename(columns={
                        'sum': f'{num_col}_sum',
                        'mean': f'{num_col}_mean',
                        'std': f'{num_col}_std',
                        'median': f'{num_col}_median'
                })
                # Note: pandas .agg automatically names columns, but we keep it explicit.
                return result

        def _aggregate_numeric_cols(self, df_grouped):
                result = df_grouped.apply(
                        lambda group: pd.Series({
                                'Total_Sales': group['Sales'].sum(),
                                'Total_Quantity': group['Quantity'].sum(),
                                'Total_Profit': group['Profit'].sum(),
                                'Discount_Amount': (group['Sales'] * group['Discount']).sum(),
                                'Sales_Per_Item': group['Sales'].sum() / group['Quantity'].sum(),
                                'Profit_Per_Item': group['Profit'].sum() / group['Quantity'].sum()
                        })
                )
  
                return result

        def _str_get_mode(self, df_grouped, str_col: str, column_name=None) -> pd.Series:
                mode_series = df_grouped[str_col].apply(
                        lambda x: x.mode()[0] if not x.mode().empty else np.nan)
                if column_name == None:
                        mode_series.name = f'{str_col}_mode'
                else:
                        mode_series.name = column_name
                return mode_series

        def _str_get_entropy(self, df_grouped, str_col: str, column_name=None) -> pd.Series:
                entropy_series = df_grouped[str_col].apply(self._normalized_entropy)

                if column_name == None:
                        entropy_series.name = f'{str_col}_entropy'
                else:
                        entropy_series.name = column_name
                return entropy_series

        def _str_get_repeat(self, df_grouped, str_col: str, categories: list[str]):
                # Unstack to get categories as columns, fill missing with 0
                counts_df = df_grouped[str_col].value_counts().unstack(fill_value=0)
                # Ensure all categories exist (reindex with the provided list)
                counts_df = counts_df.reindex(columns=categories, fill_value=0)
                # Rename columns to numberOfRep(cat)
                counts_df.columns = [f'{str_col}={cat}' for cat in categories]
                return counts_df

        def _normalized_entropy(self, series: pd.Series):
                n = len(series)
                if n == 0:
                        return np.nan
                counts = series.value_counts()
                k = len(counts)                                         # number of unique categories
                if k <= 1:
                        return 0.0
                probs = counts / n
                entropy = -sum(p * np.log(p) for p in probs)
                max_entropy = np.log(k)
                return entropy / max_entropy
