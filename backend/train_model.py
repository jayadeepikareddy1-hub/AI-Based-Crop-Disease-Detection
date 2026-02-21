import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ----------------------------
# 1️⃣ Dataset Path
# ----------------------------
dataset_path = "dataset/"   # Make sure your dataset is inside dataset/ folder

# ----------------------------
# 2️⃣ Image Preprocessing
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ----------------------------
# 3️⃣ Build CNN Model
# ----------------------------
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(train_data.num_classes, activation='softmax')
])

# ----------------------------
# 4️⃣ Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------
# 5️⃣ Train Model
# ----------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ----------------------------
# 6️⃣ Save Model
# ----------------------------
os.makedirs("model", exist_ok=True)
model.save("model/cnn_model")

print("Model training completed and saved successfully!")
