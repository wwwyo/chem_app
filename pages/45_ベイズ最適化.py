from pathlib import Path
import sys
import streamlit as st
import pandas as pd
# 親のmodule群をimportできるように
sys.path.append(str(Path(__file__).resolve().parent.parent))
from components.DownloadLink import DownloadLink
from components import Divider
from modules.BayesianOptimization2 import BayesianOptimization, KernelList

st.title('ベイズ最適化')

st.subheader('1. トレーニングデータ アップロード')
train_csv = st.file_uploader(label='train', type='csv', label_visibility='hidden', key='train')
train_df = pd.DataFrame()

st.subheader('2. 予測データ アップロード')
predicted_csv = st.file_uploader(label='predicted', type='csv', label_visibility='hidden', key='predicted')
predicted_df = pd.DataFrame()

if (train_csv and predicted_csv):
    train_df = pd.read_csv(train_csv)
    st.write(f'トレーニングデータのサイズ: {train_df.shape}')
    st.dataframe(train_df)

    predicted_df = pd.read_csv(predicted_csv)
    st.write(f'予測データのサイズ: {predicted_df.shape}')
    st.dataframe(predicted_df)

    st.subheader('3. 目的変数を選択')
    target = st.selectbox(
        key='target',
        label='target',
        label_visibility='hidden',
        options=(train_df.columns)
    )

    st.subheader('4. カーネルを選択')
    kernel = st.selectbox(
        key='model', label='kernel', label_visibility='hidden',
        options=KernelList.get_keys(),
    )

    st.subheader('5. イテレーションを設定')
    n_iter = st.number_input(label='イテレーション数', key='n_iter', value=50, min_value=1, max_value=100, step=25)

    preprocessing = st.checkbox(label='特徴量選択を行うか？', key='preprocessing', value=True)

    Divider.render()

    clicked = st.button(label='ベイズ最適化を実行', type='primary', disabled=(predicted_csv == None))
    if clicked:
        st.subheader(f'{kernel}の結果')
        bayesianOpt = BayesianOptimization(train_df, target, predicted_df, kernel, n_iter)
        best_model = bayesianOpt.optimize()
        st.write('best model is ')
        st.write(best_model)
        y_pred = bayesianOpt.predict_target(best_model)
        DownloadLink.display(y_pred.to_csv(index=False), 'ダウンロード')
        st.dataframe(y_pred)

