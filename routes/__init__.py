from flask import Blueprint
from flask_restx import Api

from routes.health_routes import api as health_ns
from routes.spreadsheet_clustering_routes import api as spreadsheet_clustering_ns

api_bp = Blueprint("api", __name__)

api = Api(
    api_bp,
    title="ClusterForm",
    version="1.0",
    description="A REST API build with Flask",
)

api.add_namespace(health_ns)
api.add_namespace(spreadsheet_clustering_ns)
