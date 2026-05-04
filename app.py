import streamlit as st

st.set_page_config(
page_title="My App",
layout="wide"
)

st.title("KNN Classifier App")
st.sidebar.title("Settings")

k = st.sidebar.slider("Select K value", 1, 15, 3)