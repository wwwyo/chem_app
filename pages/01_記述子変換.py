from pathlib import Path
import sys
import streamlit as st
import pandas as pd
# 親のmodule群をimportできるように
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.DescriptorConverter import DescriptorConverter, Methods
from components.DownloadLink import DownloadLink


st.title('記述子変換')

st.subheader('1. csvアップロード')
csv = st.file_uploader(label='csv', type='csv', label_visibility='hidden', key='csv')
df = pd.DataFrame()
if (csv):
    df = pd.read_csv(csv)
    st.dataframe(df)

st.subheader('2. 変換したいカラム名を選択')
columns = st.multiselect(
    key='columns',
    label='columns',
    label_visibility='hidden',
    options=(df.columns), default=[]
)


st.subheader('3. 記述子を選択')
methods = st.multiselect(
    key='methods', label='', label_visibility='hidden', options=Methods.get_values()
)

submitted = st.button(
    label='変換する', type='primary', disabled=(csv == None))

if submitted:
    for method in methods:
        converted_df = DescriptorConverter(df, columns, method).convert()
        converted_csv = converted_df.to_csv(index=False)
        st.subheader(f'・{method} {converted_df.shape}')
        DownloadLink.display(converted_csv,method)

        st.dataframe(converted_df)
