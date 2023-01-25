from pathlib import Path
import sys
import streamlit as st
import pandas as pd
# 親のmodule群をimportできるように
sys.path.append(str(Path(__file__).resolve().parent.parent))
from components.DownloadLink import DownloadLink
from components import Divider
from modules.ModelSearcher import ModelSearcher
from modules.Models.Models import ModelList


st.title('モデル探索')

st.subheader('1. csvアップロード')
csv = st.file_uploader(label='csv', type='csv', label_visibility='hidden', key='csv')
df = pd.DataFrame()
if (csv):
    df = pd.read_csv(csv)
    st.write(f'csvのサイズ: {df.shape}')
    st.dataframe(df)

    st.subheader('2. 目的変数を選択')
    target = st.selectbox(
        key='target',
        label='target',
        label_visibility='hidden',
        options=(df.columns)
    )
    
    st.subheader('3. モデルを選択')
    models = st.multiselect(
      key='models', label='models', label_visibility='hidden',
      options=ModelList.get_keys(),
      default=[]
    )

    isDCV = st.checkbox(label='DCVを使用するか？', key='isDCV', value=True)

    Divider.render()

    clicked = st.button(label='モデルを探索する', type='primary', disabled=(csv == None))
    if clicked:
        for model in models:
            results = ModelSearcher(df, target, model, isDCV).exec()
            col1, col2 = st.columns(2)
            st.subheader(f'{model}の結果')
            if isDCV:
                with col1:
                    st.write(f'トレーニングスコア')
                    st.write(f'r2：{results["train_r2"]}')
                    st.write(f'mae：{results["train_mae"]}')
                    st.write(f'rmse：{results["train_rmse"]}')
                with col2:
                    st.write(f'テストスコア')
                    st.write(f'r2：{results["test_r2"]}')
                    st.write(f'mae：{results["test_mae"]}')
                    st.write(f'rmse：{results["test_rmse"]}')
            else:
              st.write(f'{model} {results}')
