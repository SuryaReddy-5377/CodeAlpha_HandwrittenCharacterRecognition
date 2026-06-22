# Handwritten Character Recognition

## 📌 Project Overview
This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset. The model achieves approximately 98% accuracy on test data.

## 📊 Results
- **Model Architecture:** CNN with Conv2D, MaxPooling, and Dense layers
- **Test Accuracy:** ~98%
- **Dataset:** MNIST (70,000 images)
- **Framework:** TensorFlow/Keras
- **Best Model:** CNN

## 🛠️ Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

## 📂 Dataset
**MNIST Dataset**
- 70,000 grayscale images of handwritten digits (0-9)
- 60,000 training images
- 10,000 test images
- Each image is 28x28 pixels

## 🚀 How to Run
```bash
# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn

# Run the code
python train.py
