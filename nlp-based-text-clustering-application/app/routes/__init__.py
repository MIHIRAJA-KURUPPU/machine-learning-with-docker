#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import render_template

def register_blueprints(app):
    """Register all blueprints with the Flask application"""
    from app.routes.cluster_routes import cluster_bp
    from app.routes.sample_routes import sample_bp
    
    app.register_blueprint(cluster_bp)
    app.register_blueprint(sample_bp)
    
    # Register the index route
    @app.route('/')
    def index():
        return render_template('index.html')