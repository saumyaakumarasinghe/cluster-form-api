"""
Health check API endpoint for monitoring server status.
Provides basic health check functionality for load balancers and monitoring tools.
"""

from flask_restx import Namespace, Resource
from controller.health_controller import health_check

api = Namespace(
    "health", description="Health check endpoint for server monitoring.", path="/health"
)


@api.route("")
class Health(Resource):
    @api.response(200, "Server is healthy.")
    @api.response(500, "Server is unhealthy.")
    def get(self):
        """
        Check if the server is running and healthy.

        Returns:
            dict: Server status information
        """
        return health_check()
