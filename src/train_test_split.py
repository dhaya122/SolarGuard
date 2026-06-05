from sklearn.model_selection import train_test_split
from src.data_loader import load_images



X, y, label_map = load_images("dataset")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)