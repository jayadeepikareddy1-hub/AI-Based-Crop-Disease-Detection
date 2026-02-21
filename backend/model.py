import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
MODEL_PATH = os.path.join("model", "cnn_model.h5")
model = load_model(MODEL_PATH)

# Class labels (change according to your dataset)
class_labels = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Spot"
]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class
