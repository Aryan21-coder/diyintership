# ==========================================
# AI Movie Review Sentiment Analyzer
# ==========================================

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page title
st.set_page_config(page_title="AI Sentiment Analyzer")

# Load trained model
model = load_model("imdb_sentiment_model.h5")

# Load IMDB dictionary
word_index = imdb.get_word_index()

# Function for prediction
def predict_sentiment(review_text):

    words = review_text.lower().split()

    encoded_review = []

    for word in words:

        if word in word_index:
            encoded_review.append(word_index[word] + 3)

        else:
            encoded_review.append(2)

    # Padding
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200,
        padding='post'
    )

    # Prediction
    prediction = model.predict(padded_review, verbose=0)

    prediction_value = prediction[0][0]

    return prediction_value


# ==========================================
# Streamlit UI
# ==========================================

st.title("🎬 AI Movie Review Sentiment Analyzer")

st.write("Type a movie review below and click predict.")

# User input
review = st.text_area(
    "Movie Review",
    placeholder="Example: This movie was amazing and emotional"
)

# Predict button
if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a movie review.")

    else:

        # Get prediction
        prediction_value = predict_sentiment(review)

        # Show raw value
        st.write("Prediction Score:", round(prediction_value, 2))

        # Show result
        if prediction_value > 0.5:

            st.success("Positive Review 😊")

        else:

            st.error("Negative Review 😞")


# Footer
st.write("Streamlit + TensorFlow AI App Running Successfully 🚀")