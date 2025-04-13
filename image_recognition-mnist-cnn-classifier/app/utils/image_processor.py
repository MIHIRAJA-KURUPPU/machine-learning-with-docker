import numpy as np
from PIL import Image
import base64
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageProcessor:
    """Image processing utilities for handling MNIST image preprocessing, augmentation, and thumbnail generation."""
    
    @staticmethod
    def preprocess_image(image_file, invert=True, center_and_scale=True):
        """Process uploaded image file to correct format for MNIST model."""
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
            
            # Apply thresholding to make lines clearer
            img_array = (img_array > 128).astype(np.uint8) * 255
            
            # Center and scale the digit if requested
            if center_and_scale:
                img_array = ImageProcessor.center_and_scale_digit(img_array)
            
            # Reshape to add channel dimension (28, 28, 1) and normalize to range [0, 1]
            processed = img_array.reshape(28, 28, 1).astype('float32') / 255.0
            
            return processed
            
        except Exception as e:
            raise ValueError(f"Error during image preprocessing: {e}")

    @staticmethod
    def center_and_scale_digit(img_array):
        """Center and scale the digit within the image."""
        # Find bounding box of the digit
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        
        # If the image is empty (all black), return as is
        if not np.any(rows) or not np.any(cols):
            return img_array
        
        # Find the boundaries
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract the digit
        digit = img_array[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate the size and center of the new image
        h, w = digit.shape
        max_dim = max(h, w)
        target_size = int(0.8 * 28)  # Use 80% of the 28x28 image
        
        # Scale the digit
        scale = target_size / max_dim
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Create a blank target image
        target = np.zeros((28, 28), dtype=np.uint8)
        
        # Calculate position to place the digit
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        
        # Resize the digit and place it in the center
        from PIL import Image
        digit_img = Image.fromarray(digit)
        digit_img = digit_img.resize((new_w, new_h))
        digit_resized = np.array(digit_img)
        
        # Place the digit in the center of the target image
        target[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        return target
    

    
    @staticmethod
    def generate_augmented_versions(image_array, count=5):
        """Generate slightly modified versions of the image for additional predictions."""
        try:
            augmented = []
            
            # Reshape to (1, 28, 28, 1) for the ImageDataGenerator
            image_array = image_array.reshape(1, 28, 28, 1)
            
            # Set up the ImageDataGenerator for augmentation
            datagen = ImageDataGenerator(
                rotation_range=15,  # Random rotation between -15 and 15 degrees
                width_shift_range=0.1,  # Shift the image horizontally by 10%
                height_shift_range=0.1,  # Shift the image vertically by 10%
                zoom_range=0.1,  # Zoom in or out by 10%
                shear_range=0.1,  # Shear the image by 10%
                fill_mode='nearest'  # Fill empty pixels after transformations
            )
            
            # Generate augmented versions
            augmented_iter = datagen.flow(image_array, batch_size=1)
            
            # Collect augmented images
            for _ in range(count):
                augmented.append(next(augmented_iter)[0].reshape(28, 28, 1))
            
            return augmented
        
        except Exception as e:
            raise ValueError(f"Error during image augmentation: {e}")


