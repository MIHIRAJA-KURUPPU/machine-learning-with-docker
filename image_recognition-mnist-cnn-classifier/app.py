import os
import base64
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from flasgger import Swagger
from config import Config
from models.mnist_model import MNISTModel
from utils.image_processor import ImageProcessor
from utils.image_processor_for_images import ImageProcessor_for_images
import traceback

app = Flask(__name__)
app.config.from_object(Config)
swagger = Swagger(app)

# Initialize model
model = MNISTModel(model_path=Config.MODEL_PATH)

# Create model file if it doesn't exist
if not os.path.exists(Config.MODEL_PATH):
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    print("Training new model...")
    model.train(save_path=Config.MODEL_PATH)
else:
    model.load()

@app.route('/')
def index():
    """Render main application page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image file upload prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        processed_image = ImageProcessor_for_images.preprocess_image_for_images(file)

        if isinstance(processed_image, np.ndarray):
            if processed_image.shape == (28, 28):
                processed_image = processed_image.reshape(1, 28, 28, 1).astype(np.float32)
            elif processed_image.shape == (28, 28, 1):
                processed_image = processed_image.reshape(1, 28, 28, 1).astype(np.float32)
            elif processed_image.shape != (1, 28, 28, 1):
                raise ValueError(f"Unexpected image shape: {processed_image.shape}")
        else:
            raise TypeError("Processed image is not a NumPy array")

        prediction = model.predict(processed_image)
        file.seek(0)
        image_data = ImageProcessor_for_images.get_image_thumbnail_for_images(file)

        # Return JSON always (recommended)
        return jsonify(prediction)

    except Exception as e:
        app.logger.error(f"Error in /predict: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """Handle canvas drawing prediction"""
    data = request.get_json()
    
    if not data or 'image_data' not in data:
        app.logger.error('No image data in the request')
        return jsonify({'error': 'No image data in the request'}), 400
    
    try:
        # Process the base64 image data
        image_data = data['image_data']
        # Remove the data URI scheme prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 data
        decoded = base64.b64decode(image_data)
        
        # Create a file-like object
        from io import BytesIO
        image_file = BytesIO(decoded)
        
        # Process the image with improved preprocessing
        processed_image = ImageProcessor.preprocess_image(image_file, invert=True, center_and_scale=True)
        
        # Ensure the image is reshaped for prediction (28, 28, 1)
        processed_image = processed_image.reshape(1, 28, 28, 1)
        
        # Get prediction from the model
        prediction = model.predict(processed_image)
        
        # Generate augmented versions and get consensus
        augmented = ImageProcessor.generate_augmented_versions(processed_image)
        augmented_predictions = []
        
        for aug_img in augmented:
            aug_img = aug_img.reshape(1, 28, 28, 1)
            aug_pred = model.predict(aug_img)
            augmented_predictions.append(aug_pred)
        
        # Include these predictions in response
        prediction['augmented'] = [
            {'digit': p['digit'], 'confidence': p['confidence']} 
            for p in augmented_predictions
        ]
        
        # Add a consensus prediction based on augmented images
        all_predictions = [prediction] + augmented_predictions
        digit_counts = {}
        for p in all_predictions:
            digit = p['digit']
            digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        consensus_digit = max(digit_counts.items(), key=lambda x: x[1])[0]
        prediction['consensus'] = consensus_digit
        
        # Return the prediction results
        return jsonify(prediction)
            
    except Exception as e:
        # Log the full stack trace for better debugging
        error_message = str(e)
        error_trace = traceback.format_exc()
        app.logger.error(f"Error in /predict_canvas: {error_message}")
        app.logger.error(f"Stack trace: {error_trace}")
        
        return jsonify({'error': error_message}), 500

    
@app.route('/train', methods=['GET'])
def train_model():
    """Train a new model (for admin use)"""
    try:
        # Check for a secret key (simple authentication)
        secret = request.args.get('secret')
        if secret != 'admin-secret-key':  # This should be more secure in production
            app.logger.warning('Unauthorized access attempt')
            return jsonify({'error': 'Unauthorized'}), 401
            
        epochs = int(request.args.get('epochs', 10))
        batch_size = int(request.args.get('batch_size', 32))
        
        # Train the model
        results = model.train(epochs=epochs, batch_size=batch_size, save_path=Config.MODEL_PATH)
        
        app.logger.info(f"Model trained successfully with accuracy: {results['accuracy']}")
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': results['accuracy'],
            'error': results['error']
        })
    except Exception as e:
        app.logger.error(f"Error in /train: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize/<int:digit>')
def visualize_digit(digit):
    """Visualize examples of a specific digit from the dataset"""
    if digit < 0 or digit > 9:
        app.logger.warning(f"Invalid digit {digit} requested for visualization")
        return jsonify({'error': 'Invalid digit. Must be between 0-9'}), 400
        
    try:
        from keras.datasets import mnist
        import random
        import base64
        from io import BytesIO
        from PIL import Image
        
        # Load dataset
        (X_train, y_train), (_, _) = mnist.load_data()
        
        # Find examples of the requested digit
        indices = [i for i, y in enumerate(y_train) if y == digit]
        
        # Pick random samples (up to 9)
        sample_size = min(9, len(indices))
        samples = random.sample(indices, sample_size)
        
        # Convert samples to base64 images
        image_data = []
        for idx in samples:
            img = X_train[idx]
            im = Image.fromarray(img)
            buffered = BytesIO()
            im.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data.append(f"data:image/png;base64,{img_str}")
        
        return render_template('visualize.html', digit=digit, images=image_data)
    except Exception as e:
        app.logger.error(f"Error in /visualize/{digit}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get information about the current model"""
    if model.model is None:
        model.load()
        
    if model.model is None:
        app.logger.error('Model failed to load')
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get model summary
    stringlist = []
    model.model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    
    return jsonify({
        'model_path': model.model_path,
        'model_summary': model_summary,
        'model_layers': [layer.name for layer in model.model.layers]
    })

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
