import pandas as pd
from modules.Models.Models import ModelList, ModelRegression

class ModelRegressor():
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, model: ModelList, preprocessing):
        self.train_df = train_df
        self.train_X = train_df.drop(target, axis=1)
        self.train_y = train_df[target]
        self.test_df = test_df
        self.target = target
        self.model_name = model
        self.model_wrapper = ModelList[model].value
        self.preprocessing = preprocessing

    def exec(self):
        model = ModelRegression(self.train_X, self.train_y, self.test_df, self.model_wrapper, self.preprocessing)
        return model.predict()