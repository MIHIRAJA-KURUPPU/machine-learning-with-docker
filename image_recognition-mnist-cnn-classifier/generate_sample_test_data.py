import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
from PIL import Image
from tensorflow.keras.datasets import mnist
import numpy as np

# Create the folder if it doesn't exist
output_dir = "sample_test_images"
os.makedirs(output_dir, exist_ok=True)

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Generate and save 10 random images
for i in np.random.randint(0, 10000 + 1, 10):
    arr2im = Image.fromarray(X_train[i])
    image_path = os.path.join(output_dir, f"{i}.png")
    arr2im.save(image_path, "PNG")
