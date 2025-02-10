from flask_restx import Namespace, Resource
from controller.health_controller import health_check

api = Namespace("health", description="Health operations")


class Health(Resource):
    def get(self):
        return health_check()


api.add_resource(Health, "")
