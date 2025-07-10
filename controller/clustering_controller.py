from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import (
    INVALID_LINK,
    DATA_NOT_FOUND,
    INVALID_REQUEST_BODY_PROPERTIES,
)

from services.spreadsheet_services import SpreadsheetService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
from services.image_service import ImageService


def cluster_form(
    link: str,
    column: str,
    spreadsheet_service: SpreadsheetService,
    kclustering_service: KClusteringService,
    data_prep_service: DataPreparationService,
    image_service: ImageService,
):
    try:
        # Validate the request body properties
        if not link or not column:
            return error_response(INVALID_REQUEST_BODY_PROPERTIES)

        if "http" not in link:
            return error_response(INVALID_LINK)

        # Extract the spreadsheet ID
        spreadsheet_id = spreadsheet_service.extract_spreadsheet_id(link)
        if spreadsheet_id is None:
            return error_response(INVALID_LINK)
        print(f"Spreadsheet ID: {spreadsheet_id}")

        try:
            print(f"Spreadsheet range: {column + ':' + column}")
            # Fetch data from the spreadsheet
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
            prepared_df = data_prep_service.prepare_column_for_clustering(
                df, target_column
            )
            prepared_df = prepared_df.iloc[1:].reset_index(drop=True) # Remove the first element

            # Convert the column to a list before passing to clustering service
            feedback_list = prepared_df[target_column].tolist()
            print(f"Feedback array length: {len(feedback_list)}")

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
            print(f"Feedback array length: {len(feedback_list)}")

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
                try:
                    shortened_base64, full_base64 = (
                        image_service.compressed_image_from_base64(visualization_base64)
                    )
                    if full_base64:  # Check if the conversion was successful
                        response_data["visualization"] = (
                            full_base64  # Use the full base64 string, not shortened
                        )
                        print(f"Visualization added. Preview: {shortened_base64}")
                    else:
                        print("Failed to process visualization image")
                except Exception as e:
                    print(f"Error adding visualization: {e}")

            return success_response(response_data)

        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")


def categorizing(
    spreadsheet_link: str,
    form_link: str,
    spreadsheet_service: SpreadsheetService,
    clustering_service: KClusteringService,
    data_prep_service: DataPreparationService,
):
    try:
        # Validate the request body properties
        if not spreadsheet_link:
            return error_response(INVALID_REQUEST_BODY_PROPERTIES)
        if "http" not in spreadsheet_link:
            return error_response(INVALID_LINK)

        # Extract the spreadsheet ID
        spreadsheet_id = spreadsheet_service.extract_spreadsheet_id(spreadsheet_link)
        if spreadsheet_id is None:
            return error_response(INVALID_LINK)
        print(f"Spreadsheet ID: {spreadsheet_id}")

        # Fetch data from the spreadsheet
        spreadsheet_data = spreadsheet_service.fetch_spreadsheet_data(
            spreadsheet_id, "A1:Z"
        )
        if not spreadsheet_data:
            return error_response(DATA_NOT_FOUND)

        # Convert to DataFrame
        df = spreadsheet_service.convert_to_dataframe(
            spreadsheet_data, "A1:Z"
        )

        # Prepare data using the first column (as an example)
        target_column = df.columns[0] if len(df.columns) > 0 else "Value"
        prepared_df = data_prep_service.prepare_column_for_clustering(
            df, target_column
        )
        prepared_df = prepared_df.iloc[1:].reset_index(drop=True)
        feedback_list = prepared_df[target_column].tolist()
        print(f"Feedback array length: {len(feedback_list)}")

        # Perform clustering
        clustering_results = clustering_service.advanced_clustering(feedback_list)

        # Prepare the response
        response_data = {
            "message": "Clustering operation completed successfully.",
            "optimal_clusters": int(clustering_results["optimal_clusters"]),
        }
        return success_response(response_data)

    except Exception as e:
        return error_response(f"Error in categorizing process: {str(e)}")
