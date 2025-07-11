"""
Flask Application Factory for Cluster-Form API.
Creates and configures the Flask application with CORS and API routes.
"""

from flask import Flask
from flask_cors import CORS


def create_app(script_info=None):
    """
    Create and configure the Flask application.

    Args:
        script_info: Optional script info for Flask CLI

    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    app.config.from_object("config.DevelopmentConfig")

    # Register API blueprint
    from routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api")

    return app
