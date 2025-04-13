import base64
import traceback
import numpy as np
from io import BytesIO
from flask import Blueprint, jsonify, request, current_app
from app import model
from app.utils.image_processor import ImageProcessor
from app.utils.image_processor_for_images import ImageProcessor_for_images

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['POST'])
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
        current_app.logger.error(f"Error in /predict: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500


@prediction_bp.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """Handle canvas drawing prediction"""
    data = request.get_json()
    
    if not data or 'image_data' not in data:
        current_app.logger.error('No image data in the request')
        return jsonify({'error': 'No image data in the request'}), 400
    
    try:
        # Process the base64 image data
        image_data = data['image_data']
        # Remove the data URI scheme prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 data
        decoded = base64.b64decode(image_data)
        
        # Create a file-like object
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
        current_app.logger.error(f"Error in /predict_canvas: {error_message}")
        current_app.logger.error(f"Stack trace: {error_trace}")
        
        return jsonify({'error': error_message}), 500