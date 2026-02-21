import streamlit as st
import os
from backend.model import predict_disease
from PIL import Image

st.title("🌿 AI-Based Crop Disease Detection")

uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    result = predict_disease(temp_path)

    st.success(f"Predicted Disease: {result}")
