import streamlit as st
import pandas as pd

st.title('記述子変換')

# 1
st.subheader('1. csvアップロード')
csv = st.file_uploader(label='', type='csv', label_visibility='hidden')
df = None
if (csv):
    df = pd.read_csv(csv)
    st.dataframe(df)

st.subheader('2. 変換したいカラム名を選択')
selected_columns = st.multiselect(
    label='',
    label_visibility='hidden',
    options=(df.columns if df != None else [])
)

st.subheader('3. 記述子を選択', )
selected_columns = st.multiselect(
    label='', label_visibility='hidden', options=[]
)

submitted = st.button(
    label='変換する', type='secondary', disabled=(csv != None))
