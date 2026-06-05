# SolarGuard - Solar Panel Defect Detection System

SolarGuard is a deep learning-based web application that detects solar panel defects from images using a Convolutional Neural Network (CNN). The model classifies images into multiple categories such as clean, dusty, snow-covered, and physically damaged panels.

The project is built using **TensorFlow** and deployed using **Streamlit** for real-time predictions.

---

## Features

- Detects solar panel defects from images
- Deep learning CNN model for classification
- Supports multiple defect categories
- Real-time predictions via Streamlit UI
- Easy image upload and instant results

---

## Model Classes

- Bird-drop  
- Clean  
- Dusty  
- Electrical-damage  
- Physical-Damage  
- Snow-Covered  

---

## Project Structure

SolarGuard/
│── app.py
│── src/
│ ├── data_loader.py
│ ├── model.py
│ ├── train.py
│── models/
│ └── best_model.keras
│── dataset/
│── requirements.txt

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SolarGuard.git
cd SolarGuard

pip install -r requirements.txt

# Run application
streamlit run app.py


# Model training

python -m src.train

##Requirements
Python 3.10
TensorFlow
Streamlit
OpenCV
NumPy
Scikit-learn

##Deployment

This project is deployed using Streamlit Cloud for live predictions.

###Future Improvements
Improve accuracy using MobileNetV2 / transfer learning
Add dataset augmentation
Deploy with HuggingFace Spaces
Optimize model size for faster inference
 
#Author

Dhayalan

##Acknowledgements

TensorFlow
Streamlit
OpenCV

---

# AFTER THIS

Paste this into:

```text
README.md

Then commit:

git add README.md
git commit -m "added professional readme"
git push

