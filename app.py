import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

# Load the CNN model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Window setup
root = Tk()
root.title("MNIST Digit Classifier (CNN)")
root.geometry("420x520")

canvas_width = 280
canvas_height = 280

canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="black")
canvas.pack(pady=20)

# Create PIL image for drawing
image1 = Image.new("L", (canvas_width, canvas_height), color=0)
draw = ImageDraw.Draw(image1)

# Drawing function (medium thickness)
def draw_digit(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

canvas.bind("<B1-Motion>", draw_digit)

result_label = Label(root, text="Draw a digit", font=("Arial", 16))
result_label.pack(pady=10)

# ----------- CNN OPTIMIZED PREPROCESSING -------------
def preprocess(img):
    # Invert (white digit on black → MNIST format)
    img = ImageOps.invert(img)

    # Resize canvas to 28×28 directly (NO cropping)
    img = img.resize((28, 28))

    # Convert to numpy
    img_arr = np.array(img).astype("float32") / 255.0

    # Threshold small noise
    img_arr[img_arr < 0.2] = 0

    # Reshape for CNN
    img_arr = img_arr.reshape(1, 28, 28, 1)

    return img_arr
# ------------------------------------------------------

def predict_digit():
    img_arr = preprocess(image1)

    prediction = model.predict(img_arr)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}", fg="green")

predict_btn = Button(root, text="Predict Digit", command=predict_digit, font=("Arial", 14))
predict_btn.pack(pady=10)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, canvas_width, canvas_height), fill=0)
    result_label.config(text="Draw a digit", fg="black")

clear_btn = Button(root, text="Clear", command=clear_canvas, font=("Arial", 12))
clear_btn.pack(pady=5)

root.mainloop()