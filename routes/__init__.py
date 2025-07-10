# This file sets up the main API blueprint and registers all API namespaces for the application.
# It organizes the API structure and documentation using Flask-RESTX.

from flask import Blueprint
from flask_restx import Api

from routes.health_routes import api as health_ns
from routes.form_clustering_routes import api as form_clustering_ns

# Create a Flask blueprint for the API
api_bp = Blueprint("api", __name__)

# Create the main API object with metadata for documentation
api = Api(
    api_bp,
    title="ClusterForm",
    version="1.0",
    description="A REST API build with Flask",
)

# Register all namespaces each with their own endpoints
api.add_namespace(health_ns)
api.add_namespace(form_clustering_ns)
