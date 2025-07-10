# API endpoints for clustering a specific column of Google Form/Spreadsheet responses using KMeans.

from flask_restx import Namespace, Resource, fields
from flask import request
from controller.clustering_controller import cluster_form
from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
from services.image_service import ImageService

api = Namespace(
    "form",
    description="Cluster a specific column of Google Form/Sheet responses.",
    path="/form"
)

spreadsheet_link = api.model(
    "FormClusteringInput",
    {
        "link": fields.String(required=True, description="Google Spreadsheet link"),
        "column": fields.String(required=True, description="Column name or letter to cluster")
    }
)

@api.route("/clustering")
class Cluster(Resource):
    @api.expect(spreadsheet_link)
    @api.response(200, "Clustering completed.")
    @api.response(400, "Invalid input or error.")
    def post(self):
        """Cluster a specific column of Google Form/Sheet responses."""
        data = request.get_json()  # Get JSON body from request
        print(f"Request body: {data}")

        link = data.get("link")  # Spreadsheet link from request
        column = data.get("column")  # Column to cluster
        spreadsheet_service = SpreadsheetService()  # Spreadsheet service instance
        clustering_service = KClusteringService()  # KMeans clustering service
        data_prep_service = DataPreparationService()  # Data preparation service
        image_service = ImageService()  # Image service for visualizations

        # Call main clustering logic
        response, status_code = cluster_form(
            link,
            column,
            spreadsheet_service,
            clustering_service,
            data_prep_service,
            image_service,
        )
        return response, status_code  # Return API response
