import os
import logging
from flask import Flask
from flasgger import Swagger
from app.config import Config
from app.models.mnist_model import MNISTModel

# Initialize model
model = MNISTModel(model_path=Config.MODEL_PATH)

# Create model file if it doesn't exist
if not os.path.exists(Config.MODEL_PATH):
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    print("Training new model...")
    model.train(save_path=Config.MODEL_PATH)
else:
    model.load()

def create_app(config_class=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Configure Swagger
    swagger = Swagger(app)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.prediction import prediction_bp
    from app.routes.model import model_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(model_bp)
    
    return app