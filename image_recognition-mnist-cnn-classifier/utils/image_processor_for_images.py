import numpy as np
from PIL import Image
import base64
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageProcessor_for_images:
    """Image processing utilities for handling MNIST image preprocessing, augmentation, and thumbnail generation."""

    @staticmethod
    def preprocess_image_for_images(image_file: io.BytesIO, invert: bool = True) -> np.ndarray:
        """Process uploaded image file to correct format for MNIST model.
        
        Args:
            image_file (io.BytesIO): The image file to process.
            invert (bool): Whether to invert the image colors.

        Returns:
            np.ndarray: Processed image ready for prediction.
        """
        try:
            # Open image and convert to grayscale
            img = Image.open(image_file).convert('L')

            # Resize to 28x28
            img = img.resize((28, 28))

            # Convert to numpy array
            img_array = np.array(img)

            # Invert colors if needed (MNIST has white digits on black background)
            if invert:
                img_array = 255 - img_array

            # Reshape to (28, 28, 1) and normalize to [0, 1]
            processed = img_array.reshape(28, 28, 1).astype('float32') / 255.0

            return processed

        except Exception as e:
            raise ValueError(f"Error during image preprocessing: {e}")

    @staticmethod
    def get_image_thumbnail_for_images(image_file: io.BytesIO, size: tuple = (140, 140)) -> str:
        """Generate a base64 thumbnail of the image.
        
        Args:
            image_file (io.BytesIO): The image file to convert.
            size (tuple): Size of the thumbnail.

        Returns:
            str: Base64 encoded thumbnail as a data URI.
        """
        try:
            img = Image.open(image_file).convert('L')
            img = img.resize(size)

            # Convert to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return f"data:image/png;base64,{img_str}"

        except Exception as e:
            raise ValueError(f"Error during thumbnail generation: {e}")

    @staticmethod
    def generate_augmented_versions_for_images(image_array: np.ndarray, count: int = 5) -> list[np.ndarray]:
        """Generate slightly modified versions of the image for additional predictions.
        
        Args:
            image_array (np.ndarray): Original image array of shape (28, 28, 1).
            count (int): Number of augmented images to generate.

        Returns:
            list[np.ndarray]: List of augmented image arrays.
        """
        try:
            augmented = []

            # Reshape to (1, 28, 28, 1) for the ImageDataGenerator
            image_array = image_array.reshape(1, 28, 28, 1)

            # Set up the ImageDataGenerator for augmentation
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                shear_range=0.1,
                fill_mode='nearest'
            )

            # Generate augmented versions
            augmented_iter = datagen.flow(image_array, batch_size=1)

            # Collect augmented images
            for _ in range(count):
                augmented.append(next(augmented_iter)[0].reshape(28, 28, 1))

            return augmented

        except Exception as e:
            raise ValueError(f"Error during image augmentation: {e}")
