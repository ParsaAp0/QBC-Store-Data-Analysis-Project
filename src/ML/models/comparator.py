class ModelComparator:
        HIGHER_IS_BETTER = {
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "r2",
        }

        LOWER_IS_BETTER = {
                "mae",
                "mse",
                "rmse",
        }

        def compare(self, values: dict[str, float], metric: str) -> str:
                if not values:
                        raise ValueError("No values were provided.")
                if metric in self.HIGHER_IS_BETTER:
                        return max(values, key=values.get)
                if metric in self.LOWER_IS_BETTER:
                        return min(values, key=values.get)
                raise ValueError(f"Unsupported metric: {metric}")