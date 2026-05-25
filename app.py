# ==========================================
# AI Movie Review Sentiment Analyzer
# Connecting Input and Output Components
# Using Streamlit Reactive Flow
# ==========================================

import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# Load Saved ML Model
# ==========================================

model = load_model("imdb_sentiment_model.h5")

# Load IMDB dictionary
word_index = imdb.get_word_index()


# ==========================================
# Prediction Function
# ==========================================

def predict_sentiment(review_text):

    # Convert text into lowercase words
    words = review_text.lower().split()

    encoded_review = []

    # Convert words into numerical format
    for word in words:

        if word in word_index:
            encoded_review.append(word_index[word] + 3)

        else:
            encoded_review.append(2)

    # Make input length fixed
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200,
        padding='post'
    )

    # Predict sentiment
    prediction = model.predict(
        padded_review,
        verbose=0
    )

    prediction_value = prediction[0][0]

    return prediction_value


# ==========================================
# Streamlit User Interface
# ==========================================

st.title("🎬 AI Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review and click the button "
    "to get AI prediction."
)

# ==========================================
# Input Component
# ==========================================

review = st.text_area(
    "Movie Review Input",
    placeholder="Example: This movie was amazing and emotional"
)

# ==========================================
# Reactive Prediction Flow
# ==========================================

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        # ==========================================
        # Connect Input with Prediction Function
        # ==========================================

        prediction_value = predict_sentiment(review)

        # Calculate confidence scores
        positive_score = prediction_value * 100
        negative_score = (1 - prediction_value) * 100

        # ==========================================
        # Output Components
        # ==========================================

        st.subheader("Prediction Result")

        # Display prediction label
        if prediction_value > 0.5:

            st.success("Positive Review 😊")

        else:

            st.error("Negative Review 😞")

        # Display confidence scores
        st.write(
            f"Positive Confidence: "
            f"{positive_score:.2f}%"
        )

        st.write(
            f"Negative Confidence: "
            f"{negative_score:.2f}%"
        )

        # ==========================================
        # Bar Plot Output
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

        # Display chart
        st.pyplot(fig)


# Footer
st.write(
    "✅ Input and Output Components Connected "
    "Successfully Using Streamlit Reactive Flow"
)