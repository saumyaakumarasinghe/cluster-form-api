from flask import Blueprint
from flask_restx import Api

from src.routes.common_routes import api as common_ns
# from src.routes.link_routes import api as link_ns

api_bp = Blueprint("api", __name__)

api = Api(api_bp, title="ClusterForm",  version='1.0', description="A REST API build with Flask")

api.add_namespace(common_ns)
# api.add_namespace(link_ns)