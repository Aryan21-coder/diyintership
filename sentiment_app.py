import streamlit as st

st.title("NEW AI APP")

review = st.text_area("Enter Review")

if st.button("Predict"):

    st.success("App Working Successfully")
    