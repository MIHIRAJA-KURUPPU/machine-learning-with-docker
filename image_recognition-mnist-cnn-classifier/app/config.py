# config.py
import os
from dotenv import load_dotenv
# Load the .env file from the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)


class Config:
    """Application configuration settings"""
#     MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mnist_model.h5')
#     DEBUG = True
#     SECRET_KEY = 'mnist-secret-key-change-in-production'
#     MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size


    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default-dev-key'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    
    # Model configuration
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app', 'models', 'saved')  # Change this line
    MODEL_FILE = 'mnist_model.h5'
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
    
    # Upload configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Swagger configuration
    SWAGGER = {
        'title': 'MNIST Digit Recognition API',
        'uiversion': 3,
        'specs_route': '/swagger/',
        'description': 'API for handwritten digit recognition using MNIST model',
        'version': '1.0.0'
    }
    
    # Training configuration
    DEFAULT_EPOCHS = 10
    DEFAULT_BATCH_SIZE = 32
    
    # Admin configuration 
    ADMIN_SECRET_KEY = os.environ.get('ADMIN_SECRET_KEY') or 'admin-secret-key'
    
    # Application paths
    TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
