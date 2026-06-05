from src.data_loader import load_images

X, y, label_map = load_images("dataset")

print("Images shape:", X.shape)
print("Labels shape:", y.shape)
print("Label map:", label_map)
print("First label:", y[0])