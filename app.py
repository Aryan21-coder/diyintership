# ==========================================
# AI Movie Review Sentiment Analyzer
# Final Deployment Version
# ==========================================

import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# ==========================================
# Load Model
# ==========================================

model = load_model("imdb_sentiment_model.h5")

word_index = imdb.get_word_index()


# ==========================================
# Prediction Function
# ==========================================

def predict_sentiment(review_text):

    words = review_text.lower().split()

    encoded_review = []

    for word in words:

        if word in word_index:
            encoded_review.append(word_index[word] + 3)

        else:
            encoded_review.append(2)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200,
        padding='post'
    )

    prediction = model.predict(
        padded_review,
        verbose=0
    )

    prediction_value = prediction[0][0]

    return prediction_value


# ==========================================
# Title and Description
# ==========================================

st.title("🎬 AI Movie Review Sentiment Analyzer")

st.markdown("""
This AI application predicts whether a movie review is:

- Positive 😊
- Negative 😞

The model is trained using the IMDB Movie Review Dataset with TensorFlow and Keras.
""")


# ==========================================
# Example Reviews
# ==========================================

st.subheader("Example Reviews")

st.code("This movie was amazing and emotional")

st.code("Worst movie ever made")


# ==========================================
# User Input
# ==========================================

review = st.text_area(
    "Enter Movie Review",
    placeholder="Type your review here..."
)


# ==========================================
# Prediction
# ==========================================

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        prediction_value = predict_sentiment(review)

        positive_score = prediction_value * 100

        negative_score = (1 - prediction_value) * 100


        # Prediction Result
        st.subheader("Prediction Result")

        if prediction_value > 0.5:

            st.success("Positive Review 😊")

        else:

            st.error("Negative Review 😞")


        # Confidence Scores
        st.subheader("Confidence Scores")

        st.write(
            f"Positive Confidence: {positive_score:.2f}%"
        )

        st.write(
            f"Negative Confidence: {negative_score:.2f}%"
        )


        # ==========================================
        # Graph
        # ==========================================

        labels = ["Positive", "Negative"]

        values = [
            positive_score,
            negative_score
        ]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Confidence Percentage")

        ax.set_title("Prediction Confidence")

        st.pyplot(fig)


# ==========================================
# Footer
# ==========================================

st.markdown("---")

st.write(
    "Built using TensorFlow, Streamlit and Python 🚀"
)