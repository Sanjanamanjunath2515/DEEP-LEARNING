import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

def train_cnn():

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training CNN model...")
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    model.save("mnist_cnn_model.h5")
    print("CNN Model saved as mnist_cnn_model.h5")

if __name__ == "__main__":
    train_cnn()