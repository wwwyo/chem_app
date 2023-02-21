import abc
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.mixture import GaussianMixture
from sklearn.cross_decomposition import PLSRegression as LWPLS_Model
import lightgbm as lgb
import xgboost as xgb
# 親のmodule群をimportできるように
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.FeatureSelector import FeatureSelector
from modules.ApplicabilityDomain import ApplicabilityDomain

class ModelInterface(metaclass=abc.ABCMeta):
    IS_SCALING = True
    PREFIX = 'model__'
    param_grid = {}
    
    def get_param_grid(self):
        results = {}
        for key, value in self.param_grid.items():
            results[f'{self.PREFIX}{key}'] = value
        return results

    def new(cls, **kwargs):
        pass

class OLS(ModelInterface):
    param_grid = {}

    def new(self, **kwargs):
        return LinearRegression(**kwargs)

class PLS(ModelInterface):
    param_grid = {
        'n_components': range(5,31),
        'max_iter': [1000, 2000]
    }
    
    def new(self, **kwargs):
        return PLSRegression(**kwargs)

class LWPLS(ModelInterface):
    param_grid = {
        'n_components': np.arange(5, 31),
        'scale': [2**i for i in range(-9, 6)]}

    def new(self, **kwargs):
        return LWPLS_Model(**kwargs)

class RR(ModelInterface):
    """
    L2正則化
    """
    param_grid = {
        'alpha': np.power(2.0, np.arange(-15, 10)).tolist()
    }
    
    def new(self, **kwargs):
        return Ridge(**kwargs)
    
class LR(ModelInterface):
    """
    L1正則化
    """
    param_grid = {
        'alpha': np.power(2.0, np.arange(-15, 0)).tolist()
    }

    def new(self, **kwargs):
        return Lasso(**kwargs)

class EN(ModelInterface):
    """
    Elastic Net
    L1 + L2正則化
    """
    param_grid = {
        'alpha': np.power(2.0, np.arange(-15, 0)).tolist(),
        'l1_ratio': np.arange(0.01, 1.01, 0.01).tolist(),
    }

    def new(self, **kwargs):
        return ElasticNet(**kwargs)

class LSVR(ModelInterface):
    param_grid = {
        'epsilon':  np.power(2.0, np.arange(-10, 0)).tolist(),
        'C': np.power(2.0, np.arange(-5, 5)).tolist(),
    }

    def new(self, **kwargs):
        return LinearSVR(**kwargs)

class NSVR(ModelInterface):
    param_grid = {
        'C': np.power(2.0, np.arange(-5, 10)).tolist(),
        'epsilon': np.power(2.0, np.arange(-10, 0)).tolist(),
        'gamma': np.power(2.0, np.arange(-20, 11)).tolist(),
    }

    def new(self,**kwargs):
        return SVR(**kwargs)

class DT(ModelInterface):
    IS_SCALING = False
    param_grid = {
        'max_depth': range(2, 31),
        'min_samples_leaf': [3]
    }

    def new(self,**kwargs):
        return DecisionTreeRegressor(**kwargs)

class RF(ModelInterface):
    IS_SCALING = False
    param_grid = {
        'n_estimators': [500],
        'max_features': [0.1 * i for i in range(1, 11)]
    }

    def new(self,**kwargs):
        return RandomForestRegressor(**kwargs)

class GBDT(ModelInterface):
    IS_SCALING = False
    param_grid = {
        'n_estimators': [500],
        'max_features': [0.1 * i for i in range(1, 11)]
    }

    def new(self, **kwargs):
        return GradientBoostingRegressor(**kwargs)

class XGB(ModelInterface):
    IS_SCALING = False
    param_grid = {
        'n_estimators': [500],
        'reg_alpha': [0, 0.003, 0.1],
        # 'reg_lambda': [0.0001, 0.1],
        # 'num_leaves': [2, 3, 4, 6],
        'colsample_bytree': [0.4, 0.7, 1.0],
        'subsample': [0.4, 1.0],
    }

    def new(self, **kwargs):
        return xgb.XGBRegressor(**kwargs)

class LGB(ModelInterface):
    IS_SCALING = False
    param_grid = {
        'n_estimators': [500],
        'reg_alpha': [0, 0.003, 0.1],
        'reg_lambda': [0.0001, 0.1],
        'num_leaves': [2, 3, 4, 6],
        'colsample_bytree': [0.4, 0.7, 1.0],
        'subsample': [0.4, 1.0],
        'subsample_freq': [0, 7],
        'min_child_samples': [2, 5, 10]
    }

    def new(self, **kwargs):
        return lgb.LGBMRegressor(**kwargs)

class GMR(ModelInterface):
    param_grid = {
        'n_components': range(1,31),
        'covariance_type': ['full', 'tied', 'diag', 'spherical']
    }

    def new(self,**kwargs):
        return GaussianMixture(**kwargs)

class ModelList(Enum):
    OLS = OLS()
    PLS = PLS()
    LWPLS = LWPLS()
    Ridge = RR()
    Lasso = LR()
    EN = EN()
    LSVR = LSVR()
    NSVR = NSVR()
    DT = DT()
    RF = RF()
    GBDT = GBDT()
    XGB = XGB()
    LGB = LGB()
    GMR = GMR()

    @classmethod
    def get_keys(cls) -> List[str]:
        return [model.name for model in cls]

class ModelBuilder():
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame,  X_test: pd.DataFrame,  y_train: pd.DataFrame,  y_test: pd.DataFrame, model_wrapper: ModelList):
        self.X = X
        self.y = y

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.values
        self.y_test = y_test.values
        self.model_wrapper = model_wrapper

    def dcv(self):
        pipe = self._getPipe()
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        grid_search = GridSearchCV(pipe, self.model_wrapper.get_param_grid(), cv=kf, n_jobs=-1)
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

    def predict(self, best_params = None):
        if not best_params:
            best_params = self._cv()

        pipe = self._getPipe(best_params)
        pipe.fit(self.X_train, self.y_train)
        y_predict_train = pipe.predict(self.X_train).flatten()
        y_predict = pipe.predict(self.X_test).flatten()

        cv_results = {
            'train_r2': r2_score(self.y_train, y_predict_train),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_predict_train)),
            'train_mae': mean_absolute_error(self.y_train, y_predict_train),
            'y_predict_train': y_predict_train,
            'y_train': self.y_train,
            'test_r2': r2_score(self.y_test, y_predict),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_predict)),
            'test_mae': mean_absolute_error(self.y_test, y_predict),
            'best_params': best_params,
            'y_predict': y_predict,
            'y_test': self.y_test,
        }
        return cv_results

    def _cv(self):
        pipe = self._getPipe()
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        grid_search = GridSearchCV(pipe, self.model_wrapper.get_param_grid(), cv=kf)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        return self._removePrefix(best_params)

    def _getPipe(self, best_params=None)->Pipeline:
        steps = []
        if self.model_wrapper.IS_SCALING:
            print('scaling!!')
            steps.append(('scaler', StandardScaler()))

        if best_params:
            steps.append(('model', self.model_wrapper.new(**best_params)))
        else:
            steps.append(('model', self.model_wrapper.new()))
        return Pipeline(steps=steps)
    
    def _removePrefix(self, params: Dict[str, Any])->Dict[str, Any]:
        return {key.split('__')[1]: value for key, value in params.items()}

class ModelRegression():
    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, test_df: pd.DataFrame, model_wrapper, preprocessing):
        if preprocessing:
            print('preprocessing!!')
            pipe = Pipeline(steps=[('variance_threshold', FeatureSelector.getVarianceThreshold()), ('select_boruta', FeatureSelector.getBoruta(40))])
            pipe.fit(X_train, y_train)
            variance_features = X_train.columns[pipe.named_steps['variance_threshold'].get_support()]
            selected_features = variance_features[pipe.named_steps['select_boruta'].support_]
            self.X_train = pd.DataFrame(pipe.transform(X_train), columns=selected_features)
            self.test_df = pd.DataFrame(pipe.transform(test_df), columns=selected_features)
        else:
            self.X_train = X_train
            self.test_df = test_df
        self.y_train = y_train
        self.model_wrapper = model_wrapper
        self.preprocessing = preprocessing

    def predict(self, best_params = None):
        if not best_params:
            best_params = self._cv()

        pipe = self._getPipe(best_params)
        pipe.fit(self.X_train, self.y_train)
        train_score = pipe.score(self.X_train, self.y_train)
        y_predict = pd.Series(pipe.predict(self.test_df), name='y_predict')

        # ad
        data_density_train, data_density_pred = self.ad()

        results = {
            'df': pd.concat([y_predict, pd.Series(data_density_pred, name='ad'), self.test_df,], axis=1),
            'train_size': self.X_train.shape,
            'train_score': train_score,
        }
        return results

    def _cv(self):
        pipe = self._getPipe()
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        grid_search = GridSearchCV(pipe, self.model_wrapper.get_param_grid(), cv=kf)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        return self._removePrefix(best_params)

    def _getPipe(self, best_params=None)->Pipeline:
        steps = []
        if self.model_wrapper.IS_SCALING:
            print('scaling!!')
            steps.append(('scaler', StandardScaler()))
        if best_params:
            steps.append(('model', self.model_wrapper.new(**best_params)))
        else:
            steps.append(('model', self.model_wrapper.new()))
        return Pipeline(steps=steps)

    def _removePrefix(self, params: Dict[str, Any])->Dict[str, Any]:
        return {key.split('__')[1]: value for key, value in params.items()}

    def ad(self):
        AD = ApplicabilityDomain()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_train)
        AD.fit(X_scaled)
        data_density_train = AD.decision_function(X_scaled)
        data_density_pred = AD.decision_function(scaler.transform(self.test_df))
        return [data_density_train, data_density_pred]