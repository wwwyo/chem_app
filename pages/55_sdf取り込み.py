import pandas as pd
from rdkit import Chem
import streamlit as st
from components.DownloadLink import DownloadLink

st.title('sdfファイル読み込み')

st.subheader('sdf アップロード')
sdf = st.file_uploader(label='sdf', type='sdf', label_visibility='hidden', key='sdf')
if sdf:
    mols = [mol for mol in Chem.ForwardSDMolSupplier(sdf) if mol is not None]
    smiles = [Chem.MolToSmiles(mol) for mol in mols]
    df = pd.DataFrame({'SMILES': smiles})

    DownloadLink.display(df.to_csv(index=False), 'smiles')
    st.dataframe(df)

