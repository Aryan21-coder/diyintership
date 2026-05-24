
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

vocab_size = 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

print("Dataset Loaded Successfully")
print("Training Samples:", len(x_train))
print("Testing Samples:", len(x_test))


max_length = 200

x_train = keras.preprocessing.sequence.pad_sequences(
    x_train,
    maxlen=max_length,
    padding='post'
)

x_test = keras.preprocessing.sequence.pad_sequences(
    x_test,
    maxlen=max_length,
    padding='post'
)

print("Data Preprocessing Completed")


model = keras.Sequential([


    layers.Embedding(vocab_size, 32, input_length=max_length),

    # Learn important patterns
    layers.GlobalAveragePooling1D(),

    # Hidden layer
    layers.Dense(32, activation='relu'),

    # Output layer
    layers.Dense(1, activation='sigmoid')
])


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully")


history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)


loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", accuracy)


model.save("imdb_sentiment_model.h5")

print("\nModel Saved Successfully")
print("Saved File Name: imdb_sentiment_model.h5")



prediction = model.predict(x_test[:1])

print("\nPrediction Value:", prediction[0][0])

if prediction[0][0] > 0.5:
    print("Sentiment: Positive Review ")
else:
    print("Sentiment: Negative Review ")