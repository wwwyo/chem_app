import abc
from typing import List
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class ModelInterface(metaclass=abc.ABCMeta):
    param_grid = {}

    def new(cls, **kwargs):
        pass


class PLS(ModelInterface):
    param_grid = {
        'model__n_components': [2, 3, 4, 5, 6],
        'model__scale':[True, False],
        'model__max_iter': [500, 1000, 2000]
    }
    
    def new(self, **kwargs):
        return PLSRegression(**kwargs)

class SVR(ModelInterface):
    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__gamma': [0.001, 0.01, 0.1]
    }
    
    def new(self,**kwargs):
        return SVR(**kwargs)

class RF(ModelInterface):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    def new(self,**kwargs):
        return RandomForestRegressor(**kwargs)

class ModelList(Enum):
    PLS = PLS()
    SVR = SVR()
    RandomForest = RF()
    # XGBoost = auto()

    @classmethod
    def get_keys(cls) -> List[str]:
        return [model.name for model in cls]

class ModelBuilder():
    def __init__(self, df: pd.DataFrame, target: str, model_wrapper:ModelList):
        self.X = df.drop(columns=target)
        self.y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_wrapper = model_wrapper

    def cv(self):
        model = self.model_wrapper.new()
        pipe = Pipeline(steps=[('model', model)])
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        grid_search = GridSearchCV(pipe, model.param_grid, cv=kf)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_

    def dcv(self):
        pipe = Pipeline(steps=[('scaler', StandardScaler()),('model', self.model_wrapper.new())])
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        grid_search = GridSearchCV(pipe, self.model_wrapper.param_grid, cv=kf, n_jobs=-1)
        cv_results = cross_validate(grid_search, self.X, self.y, cv=5, return_train_score=True,
                                    scoring={'r2': make_scorer(r2_score), 
                                            'mae': make_scorer(mean_absolute_error), 
                                            'rmse': make_scorer(mean_squared_error)}
                                    )
        return {
            "test_r2": cv_results['test_r2'].mean(),
            "test_mae": cv_results['test_mae'].mean(),
            "test_rmse": np.sqrt(cv_results['test_rmse'].mean()),
            "train_r2": cv_results['train_r2'].mean(),
            "train_mae": cv_results['train_mae'].mean(),
            "train_rmse": np.sqrt(cv_results['train_rmse'].mean()),
        }

    def predict(self, best_params):
        if not best_params:
            best_params = self.cv(self.X_train, self.y_train)
        model = self.model_wrapper.new(**best_params).fit(self.X_train, self.y_train)
        y_predict = model.predict(self.X_test)
        cv_results = {
            'r2': r2_score(self.y_test, self.y_predict),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_predict)),
            'mae': mean_absolute_error(self.y_test, self.y_predict),
            'best_params': best_params,
            'y_predict': y_predict,
        }
        return cv_results
    
    