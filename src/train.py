from src.data_loader import load_images
from src.model import build_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json
import os

os.makedirs("models", exist_ok=True)

X, y, label_map = load_images("dataset")

X = X.astype("float32") / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = build_model(num_classes=len(label_map))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

model.save("models/best_model.keras")

with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)

print("Training complete. Model saved in models/best_model.keras")