# config.py
import os

class Config:
    """Application configuration"""
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'mnist_model.h5')
    DEBUG = True
    SECRET_KEY = 'mnist-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size