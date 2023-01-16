import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = "DejaVu Serif"

st.title('分析')

csv = st.file_uploader(label='csvをアップロード', type='csv', key='csv')
df = pd.DataFrame()
if csv:
    df = pd.read_csv(csv)
    st.dataframe(df)

    st.subheader('統計値')
    st.dataframe(df.describe())

    st.subheader('ヒートマップ')
    annot = st.checkbox(label='相関を表示', key='annot')
    attention = st.selectbox(label='注目', key='attention', options=['全体',*df.columns])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.set(font="DejaVu Serif") 
    sns.set_style('whitegrid')
    sns.set_palette('gray')
    if attention == '全体':
        sns.heatmap(df.corr().T, cmap="bwr", annot=annot, ax=ax)
    else:
        sns.heatmap(df.corr()[[attention]].T.sort_values(by=attention,ascending=False), cmap="bwr", annot=annot, ax=ax)
    st.pyplot(fig)