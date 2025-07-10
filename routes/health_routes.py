# API endpoint for checking server health.

from flask_restx import Namespace, Resource
from controller.health_controller import health_check

api = Namespace(
    "health",
    description="Health check endpoint.",
    path="/health"
)

@api.route("")
class Health(Resource):
    @api.response(200, "Server is healthy.")
    def get(self):
        """Check if the server is running."""
        return health_check()
