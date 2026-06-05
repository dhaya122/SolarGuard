import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

st.set_page_config(page_title="SolarGuard", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'Bird-drop',
    'Clean',
    'Dusty',
    'Electrical-damage',
    'Physical-Damage',
    'Snow-Covered'
]

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("Solar Panel Defect Detection System")

uploaded_file = st.file_uploader("Upload Solar Panel Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)

    prediction = model.predict(processed)[0]

    top_3_idx = prediction.argsort()[-3:][::-1]

    st.subheader("Prediction Results")

    for i in top_3_idx:
        st.write(f"**{class_names[i]}** : {prediction[i] * 100:.2f}%")

    best_class = top_3_idx[0]
    confidence = prediction[best_class]

    st.progress(int(confidence * 100))
    st.success(f"Final Prediction: {class_names[best_class]}")
    st.info(f"Confidence: {confidence * 100:.2f}%")