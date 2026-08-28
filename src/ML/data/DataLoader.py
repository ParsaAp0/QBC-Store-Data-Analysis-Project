from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd
import json

excel_sheets_config = "src/ML/configs/excel_sheets.json"
merges_config = "src/ML/configs/excel_merges.json"
validation_columns = "src/ML/configs/excel_columns.json"
order_column_name = "Order ID"

class DataLoader:
        def __init__(
                self,
                excel_location: str | Path,
                train_test_split_percentage: float,
                random_state: int = 42,
        ):
                self.excel_location = Path(excel_location)
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
                self._merge()
                self._split()

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
                                                                invalid_values = original[converted.isna() & original.notna()].unique().tolist()
                                                                raise ValueError(f"Invalid boolean values: {invalid_values}")

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

        def get_train_dataframe(self) -> pd.DataFrame:
                self._ensure_split()

                return self._train_dataframe.copy()

        def get_test_dataframe(self) -> pd.DataFrame:
                self._ensure_split()

                return self._test_dataframe.copy()

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

                        dataframe = dataframe.merge(
                                self._sheets[right_name],
                                on=merge["on"],
                                how=merge["how"],
                        )

                self._dataframe = dataframe

        def _split(self) -> None:
                """
                Split the merged dataframe into train and test sets.
                """

                if self._dataframe is None:
                        raise RuntimeError(
                                "No merged dataframe exists. Call load() first."
                        )
                        
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

                # The 'groups' parameter must be the column that defines the groups
                train_idx, test_idx = next(gss.split(self._dataframe, groups=self._dataframe[order_column_name]))

                train_dataframe = self._dataframe.iloc[train_idx]
                test_dataframe = self._dataframe.iloc[test_idx]

                # train_dataframe, test_dataframe = train_test_split(
                #         self._dataframe,
                #         train_size=self.train_test_split_percentage,
                #         random_state=self.random_state,
                #         shuffle=True,
                # )

                self._train_dataframe = train_dataframe.reset_index(drop=True)
                self._test_dataframe = test_dataframe.reset_index(drop=True)

        def _ensure_split(self) -> None:
                if self._train_dataframe is None or self._test_dataframe is None:
                        raise RuntimeError(
                                "Dataset has not been loaded and split yet. Call load() first."
                        )

        @staticmethod
        def _build_log(
                errors: list[str],
                warnings: list[str],
        ) -> str:
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