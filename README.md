# 🌿 AI-Based Crop Disease Detection

## 📖 Project Overview
This project aims to detect crop diseases using a Convolutional Neural Network (CNN). 
The system allows users to upload a leaf image and predicts the disease category.

This solution helps farmers identify plant diseases early and take preventive action.

---

## 🚀 Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- CNN (Deep Learning)

---

## 📂 Project Structure

backend/  
→ train_model.py (CNN training code)  
→ model.py (Prediction logic)  

frontend/  
→ app.py (User Interface using Streamlit)  

model/  
→ model_info.txt (Model information placeholder)

dataset/  
→ dataset_info.txt (Dataset details)

screenshots/  
→ UI and output screenshots

---

## 🧠 How the System Works

1. User uploads a leaf image.
2. Image is preprocessed.
3. CNN model predicts the disease class.
4. Result is displayed on the interface.

---

## 🏋️ Model Training

The CNN model is trained using:
- Convolutional layers
- MaxPooling layers
- Dense layers
- Softmax output layer

Training is implemented in:
backend/train_model.py

Note:
Due to large file size limitations, the trained model file (cnn_model.h5) is not included in this repository.

---

## ▶️ How to Run the Project

1. Install dependencies:
   pip install -r requirements.txt

2. Run Streamlit app:
   streamlit run frontend/app.py

---

## 🔮 Future Scope
- Real-time disease detection using live camera
- Mobile application deployment
- Drone-based crop monitoring
- Cloud-based disease alert system

---

## 📊 Dataset
PlantVillage Dataset (Kaggle)

---

## 👩‍💻 Author
P. Jaya Deepika Reddy
