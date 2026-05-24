# ==========================================
# Load Saved ML Model and Predict Reviews
# ==========================================

# Import required libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# STEP 1: Load Saved Model
# ==========================================

model = load_model("imdb_sentiment_model.h5")

print("Model Loaded Successfully")


# ==========================================
# STEP 2: Load Word Dictionary
# ==========================================

word_index = imdb.get_word_index()


# ==========================================
# STEP 3: Create Prediction Function
# ==========================================

def predict_sentiment(review_text):

    # Convert text to lowercase
    words = review_text.lower().split()

    # Convert words into numbers
    encoded_review = []

    for word in words:

        if word in word_index:
            encoded_review.append(word_index[word] + 3)

        else:
            encoded_review.append(2)

    # Make review length fixed
    padded_review = pad_sequences(
        [encoded_review],
        maxlen=200,
        padding='post'
    )

    # Predict sentiment
    prediction = model.predict(padded_review)

    prediction_value = prediction[0][0]

    # Return result
    if prediction_value > 0.5:
        return f"Positive Review 😊 ({prediction_value:.2f})"

    else:
        return f"Negative Review 😞 ({prediction_value:.2f})"


# ==========================================
# STEP 4: Test the Function
# ==========================================

review = "this movie was fantastic and emotional"

result = predict_sentiment(review)

print("\nReview:", review)
print("Prediction:", result)