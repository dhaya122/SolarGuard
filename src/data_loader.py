import os
import cv2
import numpy as np


IMG_SIZE = 224

def load_images(data_dir):
    images = []
    labels = []

    class_names = sorted(os.listdir(data_dir))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for label in class_names:
        folder_path = os.path.join(data_dir, label)

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(label_map[label])

            except:
                pass

    return np.array(images), np.array(labels), label_map 