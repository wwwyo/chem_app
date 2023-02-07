import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from components.DownloadLink import DownloadLink
from components import Divider

st.title('データ加工')

csv = st.file_uploader(label='csvをアップロード', type='csv', key='csv')
df = pd.DataFrame()

if csv:
    df = pd.read_csv(csv)
    converted_df = df
    st.dataframe(df)

    Divider.render()

    empty = st.checkbox(label='欠損値がある行を削除', key='empty')
    if empty: 
        converted_df.dropna(inplace=True)

    Divider.render()

    category_columns = st.multiselect(label='カテゴリ値をダミー変数に変換', key='category', options=converted_df.columns, default=[])
    converted_df = pd.get_dummies(df, columns=category_columns)

    Divider.render()

    remove_columns = st.multiselect(label='不要なカラムを削除', key='remove', options=converted_df.columns, default=[])
    converted_df = converted_df.drop(columns=remove_columns)
    
    Divider.render()

    csv_addition = st.file_uploader(label='csv列を追加する', type='csv', key='csv_addition')
    if (csv_addition):
        df_addition = pd.read_csv(csv_addition)
        converted_df = pd.concat([df, df_addition], axis=1)

    Divider.render()

    # 表示
    st.subheader(f'加工済みデータ {converted_df.shape}')
    DownloadLink.display(converted_df.to_csv(index=False), '加工済み')
    st.dataframe(converted_df)