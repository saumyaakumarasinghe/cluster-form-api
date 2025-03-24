from flask_restx import Namespace, Resource, fields
from flask import request
from controller.clustering_controller import categorizing
from services.google_services import SpreadsheetService
from services.clustering_services import ClusteringService
from services.data_preparation_service import DataPreparationService

api = Namespace("cluster", description="Clustering operations")

# Validate and describe the expected input
clustering_input = api.model(
    "ClusteringInput",
    {
        "spreadsheetLink": fields.String(
            required=True, description="Google Spreadsheet link"
        ),
        "formLink": fields.String(
            required=False, description="Optional Google Form link"
        ),
        "accuracyRequired": fields.Boolean(
            required=True, description="Enforce high accuracy"
        ),
    },
)


# Resource class representing the /clustering endpoint
class Cluster(Resource):

    @api.expect(clustering_input)
    def post(self):
        # Extract values from the request
        data = request.get_json()
        spreadsheet_link = data.get("spreadsheetLink")
        form_link = data.get("formLink")
        accuracy_required = data.get("accuracyRequired", False)

        # Services Initialization
        spreadsheet_service = SpreadsheetService()
        clustering_service = ClusteringService()
        data_prep_service = DataPreparationService()

        # Calling the Clustering Logic
        response, status_code = categorizing(
            spreadsheet_link,
            form_link,
            accuracy_required,
            spreadsheet_service,
            clustering_service,
            data_prep_service,
        )

        return response, status_code


api.add_resource(Cluster, "/")
