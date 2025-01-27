from flask_restx import Namespace, Resource

api = Namespace("common", description="Common operations")


class Common(Resource):
    def get(self):
        return {"status": "OK", "message": "Server is healthy"}


api.add_resource(Common, "")
