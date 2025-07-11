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
    path="/form",
)

# Input model
form_clustering_input = api.model(
    "FormClusteringInput",
    {
        "link": fields.String(
            required=True, description="Google Spreadsheet link to cluster"
        ),
        "column": fields.String(
            required=True,
            description="Column name or letter to cluster (e.g., 'B' or 'Feedback')",
        ),
    },
)

# Analytics cluster info model
analytics_cluster = api.model(
    "AnalyticsCluster",
    {
        "cluster": fields.Integer(description="Cluster number (0-based)"),
        "count": fields.Integer(description="Number of records in this cluster"),
        "percentage": fields.Float(
            description="Percentage of total records in this cluster"
        ),
    },
)

# Analytics model
analytics_model = api.model(
    "Analytics",
    {
        "total_records": fields.Integer(
            description="Total number of records clustered"
        ),
        "clusters": fields.List(
            fields.Nested(analytics_cluster), description="List of cluster analytics"
        ),
    },
)

# Output model
form_clustering_output = api.model(
    "FormClusteringOutput",
    {
        "message": fields.String(description="Status message for the operation"),
        "optimal_clusters": fields.Integer(description="Number of clusters found"),
        "link": fields.String(
            description="Link to the updated Google Sheet with cluster labels"
        ),
        "visualization": fields.String(
            description="Base64-encoded PNG image of the cluster pie chart"
        ),
        "analytics": fields.Nested(
            analytics_model,
            description="Analytics and summary statistics for the clustering",
        ),
    },
)


@api.route("/clustering")
class Cluster(Resource):
    @api.expect(form_clustering_input, validate=True)
    @api.response(200, "Clustering completed.", form_clustering_output)
    @api.response(400, "Invalid input or error.")
    @api.doc(
        description="Cluster a specific column of Google Form/Sheet responses using KMeans. "
        "Returns the number of clusters, a link to the updated sheet, a visualization image, and analytics."
    )
    def post(self):
        """
        Cluster a specific column of Google Form/Sheet responses.

        **Request Example:**
        {
            "link": "https://docs.google.com/spreadsheets/d/your-id",
            "column": "B"
        }

        **Response Example:**
        {
            "message": "Clustering operation completed successfully.",
            "optimal_clusters": 3,
            "link": "https://docs.google.com/spreadsheets/d/your-id/edit",
            "visualization": "data:image/png;base64,iVBOR...",
            "analytics": {
                "total_records": 61,
                "clusters": [
                    {"cluster": 0, "count": 58, "percentage": 95.1},
                    {"cluster": 1, "count": 2, "percentage": 3.3},
                    {"cluster": 2, "count": 1, "percentage": 1.6}
                ]
            }
        }
        """
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
