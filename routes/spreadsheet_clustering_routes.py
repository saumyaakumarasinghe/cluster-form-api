from flask_restx import Namespace, Resource, fields
from flask import request
from controller.spreadsheet_clustering_controller import cluster_spreadsheet
from services.spreadsheet_services import SpreadsheetService
from services.clustering_services import ClusteringService
from services.data_preparation_service import DataPreparationService


api = Namespace("link", description="Link operations")

spreadsheet_link = api.model(
    "link", {"link": fields.String(required=True, description="link string")}
)


class Link(Resource):

    @api.expect(spreadsheet_link)
    def post(self):
        data = request.get_json()
        link = data.get("link")
        spreadsheet_service = SpreadsheetService()
        clustering_service = ClusteringService()
        data_prep_service = DataPreparationService()

        response, status_code = cluster_spreadsheet(
            link, spreadsheet_service, clustering_service, data_prep_service
        )

        return response, status_code


api.add_resource(Link, "")
