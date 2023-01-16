import abc
from typing import List
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Avalon import pyAvalonTools
from mordred import Calculator, descriptors
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdMHFPFingerprint
from enum import Enum


class ConverterInterface(metaclass=abc.ABCMeta):
    BIT = 2048

    @abc.abstractmethod
    def convert(self, column: str, mols: list) -> pd.DataFrame:
        raise NotImplementedError()


class Morgan(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, self.BIT)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints,column)
    
    def __buildDataFrame(self, fingerprints: list, column: str):
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class MorganFeature(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, self.BIT, useFeatures=True)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class Maccs(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in AllChem.GetMACCSKeysFingerprint(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class RDKit(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in Chem.RDKFingerprint(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class MinHash(ConverterInterface):
    def __init__(self):
        self.encoder = rdMHFPFingerprint.MHFPEncoder()

    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in self.encoder.EncodeMol(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class Avalon(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in pyAvalonTools.GetAvalonFP(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class Atom(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [
                x for x in Pairs.GetAtomPairFingerprintAsBitVect(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class Donor(ConverterInterface):
    def convert(self, column, mols):
        fingerprints = []
        for mol in mols:
            fingerprint = [x for x in GetBPFingerprint(mol)]
            fingerprints.append(fingerprint)
        return self.__buildDataFrame(fingerprints, column)
    
    def __buildDataFrame(self, fingerprints: list, column: str) -> pd.DataFrame:
        column_names = list(
            map(lambda i: f'{i}_{column}', range(len(fingerprints[0]))))
        return pd.DataFrame(fingerprints, columns=column_names)


class Mordred(ConverterInterface):
    def __init__(self):
        self.calc = Calculator(descriptors, ignore_3D=True)  # 2D記述子

    def convert(self, base_column, mols):
        fingerprints = self.calc.pandas(mols, quiet=False)
        for column in fingerprints.columns:
            if fingerprints[column].dtypes == object:
                fingerprints[column] = fingerprints[column].values.astype(
                    np.float32)
        return self.__buildDataFrame(fingerprints, base_column)

    def __buildDataFrame(self, fingerprints: pd.DataFrame, column: str):
        return fingerprints.add_suffix(f'_{column}')


class Converter(Enum):
    morgan = Morgan()
    morgan_feature = MorganFeature()
    maccs = Maccs()
    rdkit = RDKit()
    minhash = MinHash()
    avalon = Avalon()
    atom = Atom()
    donor = Donor()
    mordred = Mordred()
