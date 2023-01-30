from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
import numpy as np

class ApplicabilityDomain:
    def __init__(self, nu=0.04, gamma=0.1):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X, y=None):
        self.ocsvm = OneClassSVM(nu=self.nu, gamma=self.gamma, kernel="rbf").fit(X)

    def predict(self, X):
        return self.ocsvm.predict(X)
    
    def decision_function(self, X):
        return self.ocsvm.decision_function(X)