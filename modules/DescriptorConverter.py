from pathlib import Path
import sys
from typing import List
import pandas as pd
from rdkit import Chem
from enum import auto, Enum
import math
sys.path.append(str(Path(__file__).resolve()))
from modules.Fingerprints.Fingerprints import Converter

class Methods(Enum):
    morgan = auto()
    morgan_feature = auto()
    maccs = auto()
    rdkit = auto()
    minhash = auto()
    avalon = auto()
    mordred = auto()
    
    @classmethod
    def get_values(cls):
        return [i.name for i in cls]


class DescriptorConverter:
    def __init__(self, df: pd.DataFrame, columns: List[str], method: Methods):
        self.df = df.dropna(subset=columns).reset_index(drop=True)
        self.columns = columns
        self.method = method

    def convert(self):
        for column in self.columns:
            smiles = self.df[column]
            converter = Converter[self.method].value
            fingerprints_df = converter.convert(
                column, self.__convertMols(smiles))
            removed_fingerprints_df = fingerprints_df[fingerprints_df.columns[~fingerprints_df.isnull(
            ).any()]]
            self.df = pd.concat([self.df.drop(columns=[column]), removed_fingerprints_df], axis=1)
        return self.df

    def __convertMols(self, smiles: List[str]):
        mols = []
        for smile in smiles:
            # 異常系の値はskip
            if smile in [ '-', 0]:
                smile = ''
            mols.append(Chem.MolFromSmiles(smile))
        return mols
