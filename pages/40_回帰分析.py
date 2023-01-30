from pathlib import Path
import sys
import streamlit as st
import pandas as pd
# 親のmodule群をimportできるように
sys.path.append(str(Path(__file__).resolve().parent.parent))
from components.DownloadLink import DownloadLink
from components import Divider
from modules.ModelRegressor import ModelRegressor
from modules.Models.Models import ModelList

st.title('回帰分析')

st.subheader('1. トレーニングデータ アップロード')
train_csv = st.file_uploader(label='train', type='csv', label_visibility='hidden', key='train')
train_df = pd.DataFrame()

st.subheader('2. テストデータ アップロード')
test_csv = st.file_uploader(label='test', type='csv', label_visibility='hidden', key='test')
test_df = pd.DataFrame()

if (train_csv and test_csv):
    train_df = pd.read_csv(train_csv)
    st.write(f'csvのサイズ: {train_df.shape}')
    st.dataframe(train_df)

    test_df = pd.read_csv(test_csv)
    st.write(f'csvのサイズ: {test_df.shape}')
    st.dataframe(test_df)

    st.subheader('3. 目的変数を選択')
    target = st.selectbox(
        key='target',
        label='target',
        label_visibility='hidden',
        options=(train_df.columns)
    )
    
    st.subheader('4. モデルを選択')
    model = st.selectbox(
        key='model', label='model', label_visibility='hidden',
        options=ModelList.get_keys(),
    )

    preprocessing = st.checkbox(label='特徴量選択を行うか？', key='preprocessing', value=True)

    Divider.render()

    clicked = st.button(label='回帰分析を実行', type='primary', disabled=(test_csv == None))
    if clicked:
        st.subheader(f'{model}の結果')
        results = ModelRegressor(train_df, test_df, target, model, preprocessing).exec()
        st.dataframe(results)


