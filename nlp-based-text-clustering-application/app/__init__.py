#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask
import logging
import os
from app.routes import register_blueprints
# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    # Register blueprints
    register_blueprints(app)
    
    return app