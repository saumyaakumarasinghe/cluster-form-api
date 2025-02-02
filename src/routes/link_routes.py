from flask_restx import Namespace, Resource, fields
from flask import request
from src.services.spreadsheet_services import process_spreadsheet_link

api = Namespace("link", description="Link operations")

spreadsheet_link = api.model(
    "link", {"link": fields.String(required=True, description="link string")}
)


class Link(Resource):

    @api.expect(spreadsheet_link)
    def post(self):
        data = request.get_json()
        link = data.get("link")

        # process the spreadsheet link
        response, status_code = process_spreadsheet_link(link)

        return response, status_code


api.add_resource(Link, "")
