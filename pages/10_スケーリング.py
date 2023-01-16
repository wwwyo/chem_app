import pandas as pd
import streamlit as st
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.Scaler import Scaler
from components.DownloadLink import DownloadLink

st.title('スケーリング(標準化)')

st.subheader('1. csvアップロード')
csv = st.file_uploader(label='csv', type='csv', label_visibility='hidden', key='csv')
df = pd.DataFrame()
if csv:
    df = pd.read_csv(csv)
    st.dataframe(df)

st.subheader('2. スケーリングしたいカラムを選択')
columns = st.multiselect(
    key='columns',
    label='columns',
    label_visibility='hidden',
    options=(df.columns), default=[]
)

clicked = st.button(
    label='スケーリングする', type='primary', disabled=(csv == None))

if clicked:
    scaled_df = Scaler(df, columns).scale()
    st.subheader(f'結果 {scaled_df.shape}')
    DownloadLink.display(scaled_df.to_csv(index=False), 'scaled')

    st.dataframe(scaled_df)
