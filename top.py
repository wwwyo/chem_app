import streamlit as st
from components import Bg

Bg.render()
st.markdown('''
<h1 style='font-size: 64px; font-family:serif; text-align: right;margin-top:84px; color:white;'>Chemoinfo</h1>

<div style='font-size: 12px;width: 28rem; text-align:right;margin: auto 0 auto auto; color:white;'>Introducing our new chemoinformatics app - the ultimate tool for analyzing chemical data. Streamline your research with advanced algorithms and customizable features. Get the insights you need, faster than ever before. Try it today!</div>

<p style='font-size: 34px; text-align: right;margin-top:64px;color:gold;line-height:1.3;font-family:serif;'>Begin experimenting <br/>quickly and easily.</p>
<style>
.block-container {
  max-width: 100vw !important;
}
</style>
''', unsafe_allow_html=True)
