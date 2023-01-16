from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self, df: pd.DataFrame, target_columns: List[str]):
        self.df = df.dropna().reset_index(drop=True)
        self.target_columns = target_columns

    def scale(self) -> pd.DataFrame:
        scaler = StandardScaler()
        scaler.fit(self.df[self.target_columns])
        scaled_target_df = pd.DataFrame(scaler.transform(self.df[self.target_columns]), index=self.df.index, columns=self.target_columns)
        scaled_df = pd.concat([self.df.drop(columns=self.target_columns),scaled_target_df],axis=1)
        return scaled_df

