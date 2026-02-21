import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



# Class labels (change according to your dataset)
class_labels = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Spot"
]

def predict_disease(img_path):
    # Dummy prediction for demonstration
    return "Early Blight"
    
