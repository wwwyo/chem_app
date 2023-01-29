import pandas as pd
from modules.Models.Models import ModelList, ModelBuilder

class ModelSearcher():
    def __init__(self, df: pd.DataFrame, target: str, model:ModelList, isDCV: bool, preprocessing):
        self.df = df
        self.target = target
        self.model_name = model
        self.model_wrapper = ModelList[model].value
        self.isDCV = isDCV
        self.preprocessing = preprocessing
    
    def exec(self):
        model = ModelBuilder(self.df, self.target, model_wrapper=self.model_wrapper, preprocessing=self.preprocessing)
        if (self.isDCV):
            return model.dcv()
        else:
            return model.predict()