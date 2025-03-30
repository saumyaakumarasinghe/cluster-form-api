from flask_restx import Namespace, Resource, fields
from flask import request
from controller.spreadsheet_clustering_controller import cluster_spreadsheet
from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService


api = Namespace("spreadsheet", description="Spreadsheet operations")

spreadsheet_link = api.model(
    "link", {"link": fields.String(required=True, description="link string")}
)


class Cluster(Resource):

    @api.expect(spreadsheet_link)
    def post(self):
        data = request.get_json()
        print(f"Request body: {data}")

        link = data.get("link")
        spreadsheet_service = SpreadsheetService()
        clustering_service = KClusteringService()
        data_prep_service = DataPreparationService()

        response, status_code = cluster_spreadsheet(
            link, spreadsheet_service, clustering_service, data_prep_service
        )

        return response, status_code


api.add_resource(Cluster, "/clustering")
