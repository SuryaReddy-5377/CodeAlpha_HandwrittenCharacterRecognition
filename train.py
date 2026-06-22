"""
TASK 1: Handwritten Character Recognition
CodeAlpha Internship
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("="*50)
print("🖊️ HANDWRITTEN RECOGNITION - TRAINING")
print("="*50)

# Load dataset
print("\n📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build CNN Model
print("\n🏗️ Building CNN Model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train
print("\n🚀 Training Model...")
model.fit(x_train, y_train, 
          epochs=5, 
          validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Save model
model.save('model.h5')
print("💾 Model saved as 'model.h5'")

# Test on sample images
print("\n📊 Testing on sample images...")
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.figure()
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}')
    plt.axis('off')
    plt.show()

print("\n🎉 Training Complete!")
