import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matloplib.pyplot as plt


(x_train,y_train), (x_test,y_test) = datasets.cifar10.load_data()

x_train, x_test = x_train/255.0, x_test/255.0
model = models.Sequential()
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

model.save('my_model.h5')
print("Model saved to my_model.h5")

