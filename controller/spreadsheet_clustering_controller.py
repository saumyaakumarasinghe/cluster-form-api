from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND
from constants.link_constants import SPREADSHEET_RANGE
from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
import numpy as np
import os


def cluster_spreadsheet(
    link: str,
    spreadsheet_service: SpreadsheetService,
    kclustering_service: KClusteringService,
    data_prep_service: DataPreparationService,
):
    try:
        # validate the link
        if not link:
            return error_response(EMPTY_LINK)

        if "http" not in link:
            return error_response(INVALID_LINK)

        # extract the spreadsheet ID
        spreadsheet_id = spreadsheet_service.extract_spreadsheet_id(link)
        if spreadsheet_id is None:
            return error_response(INVALID_LINK)

        try:
            # fetch data with flexible range handling
            spreadsheet_data = spreadsheet_service.fetch_spreadsheet_data(
                spreadsheet_id, SPREADSHEET_RANGE
            )
            if not spreadsheet_data:
                return error_response(DATA_NOT_FOUND)

            # convert to dataFrame
            df = spreadsheet_service.convert_to_dataframe(
                spreadsheet_data, SPREADSHEET_RANGE
            )

            # get the actual column name or index we're working with
            target_column = df.columns[0] if len(df.columns) > 0 else "Value"

            # prepare data using the correct column
            prepared_df, data_quality_metrics = (
                data_prep_service.prepare_column_for_clustering(df, target_column)
            )

            # convert the column to a list before passing to clustering service
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

            # Create a new spreadsheet with the clustering results
            try:
                # Create a dataframe with the original data and cluster labels
                result_df = prepared_df.copy()
                result_df["cluster"] = clustering_results["labels"]

                # Create a results spreadsheet
                result_spreadsheet_title = (
                    f"Clustering Results - {os.path.basename(link)}"
                )
                result_spreadsheet_id = spreadsheet_service.create_spreadsheet(
                    result_spreadsheet_title
                )

                # Write the dataframe to the new spreadsheet
                result_spreadsheet_link = (
                    spreadsheet_service.write_dataframe_to_spreadsheet(
                        result_spreadsheet_id, result_df
                    )
                )
            except Exception as sheet_error:
                print(f"Error creating results spreadsheet: {str(sheet_error)}")
                result_spreadsheet_link = None

            # Prepare the response with only what's needed for the frontend
            response_data = {
                "message": "Clustering operation completed successfully.",
                "optimal_clusters": int(clustering_results["optimal_clusters"]),
                "result_spreadsheet_link": result_spreadsheet_link,
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
            print(f"Results spreadsheet: {response_data['result_spreadsheet_link']}")

            return success_response(response_data)

        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")
