"""
TASK 1: Fast Prediction - Upload and Predict
CodeAlpha Internship
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
import urllib.request

print("="*50)
print("🖊️ FAST PREDICTION - UPLOAD & PREDICT")
print("="*50)

# Download model from GitHub
print("\n📥 Downloading trained model from GitHub...")
url = "https://raw.githubusercontent.com/SuryaReddy-5377/CodeAlpha_HandwrittenCharacterRecognition/main/model.h5"
urllib.request.urlretrieve(url, "model.h5")

# Load model
model = tf.keras.models.load_model('model.h5')
print("✅ Model loaded successfully!")

# Upload image
print("\n📤 Upload your handwritten digit image")
print("👉 Click 'Choose Files'")
uploaded = files.upload()

def predict_image(filename):
    # Load and process
    img = Image.open(filename).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = 1 - img_array  # Invert for MNIST
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    pred = model.predict(img_array, verbose=0)
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100
    
    # Show result
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f'Your Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    colors = ['red' if i == digit else 'blue' for i in range(10)]
    plt.bar(range(10), pred[0], color=colors)
    plt.title(f'Predicted: {digit} ({confidence:.1f}%)')
    plt.xlabel('Digit')
    plt.ylabel('Confidence')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*40}")
    print(f"🔢 PREDICTED: {digit}")
    print(f"📊 CONFIDENCE: {confidence:.1f}%")
    print(f"{'='*40}")

# Process all uploaded images
for filename in uploaded.keys():
    print(f"\n📂 Processing: {filename}")
    predict_image(filename)

print("\n✅ DONE!")
