from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND

from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
import numpy as np


def cluster_spreadsheet(
    link: str,
    column: str,
    spreadsheet_service: SpreadsheetService,
    kclustering_service: KClusteringService,
    data_prep_service: DataPreparationService,
):
    try:
        # Validate the link
        if not link:
            return error_response(EMPTY_LINK)

        if "http" not in link:
            return error_response(INVALID_LINK)

        # Extract the spreadsheet ID
        spreadsheet_id = spreadsheet_service.extract_spreadsheet_id(link)
        if spreadsheet_id is None:
            return error_response(INVALID_LINK)

        try:
            # Fetch data with flexible range handling
            spreadsheet_data = spreadsheet_service.fetch_spreadsheet_data(
                spreadsheet_id, column + ":" + column
            )
            if not spreadsheet_data:
                return error_response(DATA_NOT_FOUND)

            # Convert to DataFrame
            df = spreadsheet_service.convert_to_dataframe(
                spreadsheet_data, column + ":" + column
            )

            # Get the actual column name or index we're working with
            target_column = df.columns[0] if len(df.columns) > 0 else "Value"

            # Prepare data using the correct column
            prepared_df, data_quality_metrics = (
                data_prep_service.prepare_column_for_clustering(df, target_column)
            )

            # Convert the column to a list before passing to clustering service
            feedback_list = prepared_df[target_column].tolist()

            # Perform clustering
            clustering_results = kclustering_service.advanced_clustering(feedback_list)

            # Generate visualization as base64 string
            try:
                visualization_base64 = kclustering_service.visualize_clustering_results(
                    feedback_list,
                    clustering_results["labels"],
                    clustering_results["optimal_clusters"],
                    output_format="base64",
                )
            except Exception as viz_error:
                print(f"Visualization error: {str(viz_error)}")
                visualization_base64 = None

            # Add cluster labels to the existing spreadsheet
            try:
                result_df = prepared_df.copy()
                result_df["cluster"] = clustering_results["labels"]

                # Write the updated dataframe back to the same spreadsheet
                result_spreadsheet_link = (
                    spreadsheet_service.write_dataframe_to_spreadsheet(
                        spreadsheet_id, result_df
                    )
                )

                if not result_spreadsheet_link:
                    raise Exception(
                        "Failed to update spreadsheet with clustering results."
                    )
            except Exception as sheet_error:
                print(f"Error updating spreadsheet: {str(sheet_error)}")
                return error_response(f"Error updating spreadsheet: {str(sheet_error)}")

            # Prepare the response with only what's needed for the frontend
            response_data = {
                "message": "Clustering operation completed successfully.",
                "optimal_clusters": int(clustering_results["optimal_clusters"]),
                "link": result_spreadsheet_link,
            }

            # Add visualization if available
            if visualization_base64:
                response_data["visualization"] = (
                    f"data:image/png;base64,{visualization_base64}"
                )

            # Include basic metrics for logging/debugging
            print(
                f"Clustering completed with {response_data['optimal_clusters']} clusters"
            )
            print(f"Results spreadsheet: {response_data['link']}")

            return success_response(response_data)

        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")
