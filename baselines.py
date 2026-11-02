import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class ZeroRuleClassifier:
    """
    Majority Class Baseline for Classification (Building-Floor).
    Always predicts the most frequent class from the training set.
    """
    def __init__(self):
        self.majority_class = None

    def fit(self, y_train):
        self.majority_class = y_train.mode()[0]

    def predict(self, X):
        return np.array([self.majority_class] * len(X))

    def score(self, X, y_true):
        preds = self.predict(X)
        return accuracy_score(y_true, preds)


class GlobalCentroidRegressor:
    """
    Global Mean Baseline for 2D Coordinate Regression.
    Always predicts the mean longitude and latitude of the training set.
    """
    def __init__(self):
        self.mean_lat = None
        self.mean_lon = None

    def fit(self, coords_train):
        self.mean_lon = coords_train["LONGITUDE"].mean()
        self.mean_lat = coords_train["LATITUDE"].mean()

    def predict(self, X):
        preds = pd.DataFrame({
            "LONGITUDE": [self.mean_lon] * len(X),
            "LATITUDE": [self.mean_lat] * len(X)
        })
        return preds
