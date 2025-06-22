import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor

class PolynomialRANSAC:
    """
    https://stackoverflow.com/a/45351506/30470547
    """
    def __init__(self, data: np.array, degree: int = 3, **kwargs):
        self.data: np.array = data
        self.degree: int = degree
        self.predictors: np.array = None
        self.target: np.array = data[:, 2]
        self.regressor: RANSACRegressor = RANSACRegressor(**kwargs)
    
    def poly_features(self):
        self.predictors = PolynomialFeatures(self.degree).fit_transform(self.data[:, 0:2])
    
    def fit(self):
        self.regressor.fit(self.predictors, self.target)
