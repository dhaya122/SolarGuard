import numpy as np
import cv2
import tensorflow as tf
import json


model = tf.keras.models.load_model("solar_model_final.h5")


with open("label_map.json", "r") as f:
    label_map = json.load(f)


label_names = {v: k for k, v in label_map.items()}


IMG_SIZE = 224

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(img_path):
    img = preprocess_image(img_path)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)

    confidence = np.max(predictions)

    class_name = label_names[class_index]

    print("\n Prediction Result:")
    print("Class:", class_name)
    print("Confidence:", float(confidence))



if __name__ == "__main__":
    image_path = r"C:\Users\dhaya\Documents\SolarGuard\dataset\Snow-Covered\Snow (3).jpg"
    predict(image_path)
