import streamlit as st
from predict_page import show_predict_page
from info_page import show_info_page

selectbox_placeholder = st.empty()
page = selectbox_placeholder.selectbox("*Show model details*", ("Predict", "See Details"))

if page == "Predict":
    show_predict_page()
else:
    show_info_page()