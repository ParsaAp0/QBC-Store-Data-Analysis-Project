from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ..interface import grader_interface

class ClassificationGrader(grader_interface):

        def score(self, y_true, y_predict) -> dict:
                return {
                        "accuracy": accuracy_score(
                                y_true,
                                y_predict,
                        ),
                        "precision": precision_score(
                                y_true,
                                y_predict,
                                average="weighted",
                                zero_division=0,
                        ),
                        "recall": recall_score(
                                y_true,
                                y_predict,
                                average="weighted",
                                zero_division=0,
                        ),
                        "f1": f1_score(
                                y_true,
                                y_predict,
                                average="weighted",
                                zero_division=0,
                        ),
                }