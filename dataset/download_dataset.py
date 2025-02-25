#%%
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os
import cv2

# Download the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display the shapes of the data
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
#%%
output_dir = 'mnist_images'
os.makedirs(output_dir, exist_ok=True)

# Save the first 10 images as .jpg (grayscale)
for i in range(10):
    img = X_train[i]
    
    # Save the image directly in grayscale
    cv2.imwrite(os.path.join(output_dir, f"image_{i+1}.jpg"), img)
    print(f"Image {i+1} saved as grayscale JPG.")