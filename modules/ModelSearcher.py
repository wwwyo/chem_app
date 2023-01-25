import pandas as pd
from modules.Models.Models import ModelList, ModelBuilder

class ModelSearcher():
    def __init__(self, df: pd.DataFrame, target: str, model:ModelList, isDCV: bool):
        self.df = df
        self.target = target
        self.model_name = model
        self.model_wrapper = ModelList[model].value
        self.isDCV = isDCV
    
    def exec(self):
        model = ModelBuilder(self.df, self.target, self.model_wrapper)
        if (self.isDCV):
            return model.dcv()
        else:
            return model.predict()