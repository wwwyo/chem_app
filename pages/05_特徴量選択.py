import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.FeatureSelector import FeatureSelector, Methods
from components.DownloadLink import DownloadLink

st.title('特徴量選択')

st.subheader('1. csvアップロード')
csv = st.file_uploader(label='csv', type='csv', label_visibility='hidden', key='csv')
df = pd.DataFrame()
if csv:
    df = pd.read_csv(csv)
    st.dataframe(df)

st.subheader('2. 選択方法を選択')
selected_methods = st.multiselect(
    key='methods',
    label='methods',
    label_visibility='hidden',
    options=Methods.get_values(), default=Methods.get_values()
)

st.subheader('3. 文字列を含む説明変数(X)を選択')
hidden_columns = st.multiselect(
  key='hidden_columns',
  label='＊文字列のデータは計算できないため除外します。',
  options=df.columns,
  disabled=csv == None,
)

st.subheader('4. 目的変数(y)を選択')
y_column = st.selectbox(
  key='y_column',
  label='y_column',
  label_visibility='hidden',
  options=df.columns,
  disabled=csv == None,
)

clicked = st.button(label='特徴量を選択する', type='primary', disabled=(csv == None))
  
if clicked:
    selected_df = FeatureSelector(df.drop(columns=y_column), df[y_column], hidden_columns, selected_methods).select()
    st.subheader(f'結果 {selected_df.shape}')
    DownloadLink.display(selected_df.to_csv(index=False), 'selected')

    st.dataframe(selected_df)
