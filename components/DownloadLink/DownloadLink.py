import base64
import streamlit as st

def display(csv, filename):
    b64 = base64.b64encode(csv.encode('utf-8')).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.csv">ダウンロードする</a>'
    st.markdown(f"{href}", unsafe_allow_html=True)