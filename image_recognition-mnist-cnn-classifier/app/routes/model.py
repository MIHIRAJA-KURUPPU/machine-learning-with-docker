import base64
from io import BytesIO
import random
from flask import Blueprint, jsonify, request, render_template, current_app
from app import model
from app.config import Config

model_bp = Blueprint('model', __name__)

@model_bp.route('/train', methods=['GET'])
def train_model():
    """Train a new model (for admin use)"""
    try:
        # Check for a secret key (simple authentication)
        secret = request.args.get('secret')
        if secret != Config.ADMIN_SECRET_KEY:  # This should be more secure in production
            current_app.logger.warning('Unauthorized access attempt')
            return jsonify({'error': 'Unauthorized'}), 401
            
        epochs = int(request.args.get('epochs', 10))
        batch_size = int(request.args.get('batch_size', 32))
        
        # Train the model
        results = model.train(epochs=epochs, batch_size=batch_size, save_path=Config.MODEL_PATH)
        
        current_app.logger.info(f"Model trained successfully with accuracy: {results['accuracy']}")
        return jsonify({
            'message': 'Model trained successfully',
            'accuracy': results['accuracy'],
            'error': results['error']
        })
    except Exception as e:
        current_app.logger.error(f"Error in /train: {str(e)}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/visualize/<int:digit>')
def visualize_digit(digit):
    """Visualize examples of a specific digit from the dataset"""
    if digit < 0 or digit > 9:
        current_app.logger.warning(f"Invalid digit {digit} requested for visualization")
        return jsonify({'error': 'Invalid digit. Must be between 0-9'}), 400
        
    try:
        from keras.datasets import mnist
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
        current_app.logger.error(f"Error in /visualize/{digit}: {str(e)}")
        return jsonify({'error': str(e)}), 500



@model_bp.route('/model_info')
def model_info():
    """Get information about the current model"""
    if model.model is None:
        model.load()
        
    if model.model is None:
        current_app.logger.error('Model failed to load')
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




