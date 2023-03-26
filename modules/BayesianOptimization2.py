import pandas as pd
import numpy as np
from typing import List
import GPyOpt
from enum import Enum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

class KernelList(Enum):
    rbf = 'rbf' # 放射基底関数カーネル（Radial basis function kernel, RBF kernel）と定数カーネル（Constant kernel）の積
    polynomial = 'polynomial' # 多項式カーネル
    matern = 'matern' # 入力空間でのデータ点間の距離に対して異なるスケールで類似度を測る
    white = 'white' # rbfにノイズを加えたもの

    @classmethod
    def get_keys(cls) -> List[str]:
        return [kernel.name for kernel in cls]

class BayesianOptimization:
    def __init__(self, dataset: pd.DataFrame, target_variable: str, target_dataset: pd.DataFrame, kernel_type: KernelList, n_iter: int):
        self.dataset = dataset
        self.target_variable = target_variable
        self.target_dataset = target_dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataset.drop(columns=target_variable), dataset[target_variable], test_size=0.2, random_state=42)
        self.kernel_type = kernel_type
        self.n_iter = n_iter # 25 ~ 100

    def get_kernel(self, kernel_type: KernelList, length_scale: float, noise=None):
        if kernel_type == 'rbf':
            kernel = RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            kernel = Matern(length_scale=length_scale)
        elif kernel_type == 'polynomial':
            degree = 2
            kernel = (DotProduct(sigma_0=1.0) + 1.0) ** degree
        elif kernel_type == 'white':
            kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise)
        else:
            raise Exception('kernel_type is not defined')
        return kernel
        
    def objective_function(self, params):
        length_scale, noise, n_restarts_optimizer = params[0]
        kernel =  self.get_kernel(self.kernel_type, length_scale, noise)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(n_restarts_optimizer))

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        # mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # 他の指標で評価したい場合は変更
        return  r2

    def optimize(self):
        # GPyOptでベイズ最適化を設定
        bounds = [{'name': 'length_scale', 'type': 'continuous', 'domain': (0.1, 10)},
                  {'name': 'noise', 'type': 'continuous', 'domain': (1e-10, 1e-5)},
                  {'name': 'n_restarts_optimizer', 'type': 'discrete', 'domain': (0, 20)}]

        optimizer = GPyOpt.methods.BayesianOptimization(f=self.objective_function, domain=bounds)

        # 最適化の実行
        optimizer.run_optimization(max_iter=self.n_iter)

        best_params = optimizer.X[np.argmin(optimizer.Y)]
        return best_params

    def predict_target(self, best_params):
        length_scale, noise, n_restarts_optimizer = best_params

        kernel = self.get_kernel(self.kernel_type, length_scale, noise)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=int(n_restarts_optimizer))
        model.fit(self.X_train, self.y_train)
        result = model.predict(self.target_dataset, return_std=True)
        return pd.DataFrame(np.array(result).T, index=self.target_dataset.index, columns=[self.target_variable, '標準偏差'])
