from pathlib import Path
import sys
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage, rdMolDraw2D

# 親のmodule群をimportできるように
sys.path.append(str(Path(__file__).resolve().parent.parent))
from components.DownloadLink import DownloadLink
from components import Divider

st.title('smiles表示')

st.subheader('1. データ アップロード')
csv = st.file_uploader(label='csv', type='csv', label_visibility='hidden', key='csv')
df = pd.DataFrame()

if (csv):
    df = pd.read_csv(csv)
    st.write(f'csvのサイズ: {df.shape}')
    st.dataframe(df)

    st.subheader('2. smilesを選択')
    smiles_col = st.selectbox(
        key='smiles_col',
        label='smiles_col',
        label_visibility='hidden',
        options=(df.columns)
    )

    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        img = MolToImage(mol)
        st.image(img, caption=smiles,use_column_width=True)
