import pandas as pd
from modules.Models.Models import ModelList, ModelBuilder

class ModelSearcher():
    def __init__(self,  X: pd.DataFrame, y: pd.DataFrame, X_train: pd.DataFrame,  X_test: pd.DataFrame,  y_train: pd.DataFrame,  y_test: pd.DataFrame, model:ModelList, isDCV: bool):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_name = model
        self.model_wrapper = ModelList[model].value
        self.isDCV = isDCV

    def exec(self):
        model = ModelBuilder(self.X, self.y, X_train=self.X_train, X_test=self.X_test, y_train=self.y_train,y_test=self.y_test, model_wrapper=self.model_wrapper)
        if (self.isDCV):
            return model.dcv()
        else:
            return model.predict()