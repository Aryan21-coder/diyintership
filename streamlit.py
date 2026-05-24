# ==========================================
# Basic Streamlit App
# ==========================================

# Import Streamlit
import streamlit as st

# App title
st.title("My First AI Web App")

# Simple text
st.write("Hello! Streamlit is successfully installed.")

# User input
name = st.text_input("Enter your name:")

# Button action
if st.button("Submit"):

    if name != "":
        st.success(f"Welcome {name} 🚀")
    else:
        st.warning("Please enter your name")


# Slider example
age = st.slider("Select your age", 1, 100, 18)

st.write("Your age is:", age)

# Checkbox example
if st.checkbox("Show Message"):
    st.write("Streamlit setup completed successfully ✅")


# Footer
st.write("Basic Python Streamlit application is running successfully.")