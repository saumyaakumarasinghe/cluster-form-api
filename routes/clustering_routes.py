# clustering_routes.py
# API endpoints for clustering Google Spreadsheet/Form responses using KMeans.

from flask_restx import Namespace, Resource, fields
from flask import request
from controller.clustering_controller import categorizing
from services.google_services import SpreadsheetService
from services.clustering_services import ClusteringService
from services.data_preparation_service import DataPreparationService

api = Namespace(
    "cluster",
    description="Cluster Google Form/Sheet responses using KMeans.",
    path="/cluster"
)

clustering_input = api.model(
    "ClusteringInput",
    {
        "spreadsheetLink": fields.String(required=True, description="Google Spreadsheet link"),
        "formLink": fields.String(required=False, description="Google Form link (optional)")
    },
)

@api.route("/")
class Cluster(Resource):
    @api.expect(clustering_input)
    @api.response(200, "Clustering completed.")
    @api.response(400, "Invalid input or error.")
    def post(self):
        """Cluster responses from a Google Sheet or Form."""
        data = request.get_json()
        spreadsheet_link = data.get("spreadsheetLink")
        form_link = data.get("formLink")

        spreadsheet_service = SpreadsheetService()
        clustering_service = ClusteringService()
        data_prep_service = DataPreparationService()

        response, status_code = categorizing(
            spreadsheet_link,
            form_link,
            spreadsheet_service,
            clustering_service,
            data_prep_service,
        )
        return response, status_code
