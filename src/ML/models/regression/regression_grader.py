from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from ..interface import grader_interface

class RegressionGrader(grader_interface):
        def score(self, y_true, y_predict) -> dict:
                mse = mean_squared_error(
                        y_true,
                        y_predict,
                )

                return {
                        "mae": mean_absolute_error(
                                y_true,
                                y_predict,
                        ),
                        "mse": mse,
                        "rmse": np.sqrt(mse),
                        "r2": r2_score(
                                y_true,
                                y_predict,
                        ),
                }