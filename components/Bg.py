import streamlit as st

def render()->st:
    bg_img = '''
    <style>
    .stApp {
      background-image: url("https://drive.google.com/uc?export=view&id=1eG0HtX4tDht4U9fxPLe52tnxK9bfX_cM"),linear-gradient(to bottom, rgba(25,26,42,1) 30%, rgba(18,16,23,1));
      background-size: cover;
      background-repeat: no-repeat;
      opacity: 0.85;
    }
    </style>
    '''

    return st.markdown(bg_img, unsafe_allow_html=True)