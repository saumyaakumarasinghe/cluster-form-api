from flask_restx import Namespace, Resource, fields
from flask import request
from controller.clustering_controller import cluster_form
from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
from services.image_service import ImageService


api = Namespace("form", description="Form operations")

spreadsheet_link = api.model(
    "link", {"link": fields.String(required=True, description="link string")},
    "column", {"link": fields.String(required=True, description="column string")}
)


class Cluster(Resource):

    @api.expect(spreadsheet_link)
    def post(self):
        data = request.get_json()
        print(f"Request body: {data}")

        link = data.get("link")
        column = data.get("column")
        spreadsheet_service = SpreadsheetService()
        clustering_service = KClusteringService()
        data_prep_service = DataPreparationService()
        image_service = ImageService()

        response, status_code = cluster_form(
            link,
            column,
            spreadsheet_service,
            clustering_service,
            data_prep_service,
            image_service,
        )

        return response, status_code


api.add_resource(Cluster, "/clustering")
