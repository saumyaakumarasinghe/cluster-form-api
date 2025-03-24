from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import (
    EMPTY_LINK,
    INVALID_LINK,
    DATA_NOT_FOUND,
    ACCURACY_REQUIRED,
)
from constants.link_constants import SPREADSHEET_RANGE
from services.google_services import SpreadsheetService
from services.clustering_services import ClusteringService
from services.data_preparation_service import DataPreparationService


def categorizing(
    spreadsheet_link: str,
    form_link: str,
    accuracy_required: bool,
    spreadsheet_service: SpreadsheetService,
    clustering_service: ClusteringService,
    data_prep_service: DataPreparationService,
):
    try:
        # Validate the link
        if not spreadsheet_link:
            return error_response(EMPTY_LINK)

        if "http" not in spreadsheet_link:
            return error_response(INVALID_LINK)

        # Extract the spreadsheet ID
        spreadsheet_id = spreadsheet_service.extract_spreadsheet_id(spreadsheet_link)
        if spreadsheet_id is None:
            return error_response(INVALID_LINK)
        print(f"Spreadsheet ID: {str(spreadsheet_id)}")

        try:
            # Attempt metadata extraction for linked Google Form
            open_ended_column = None
            if form_link:
                open_ended_column = spreadsheet_service.get_open_ended_column_from_form(
                    form_link
                )

            # If no form link is available or metadata extraction fails, use heuristic detection
            if not open_ended_column:
                spreadsheet_data = spreadsheet_service.fetch_spreadsheet_data(
                    spreadsheet_id, SPREADSHEET_RANGE
                )
                if not spreadsheet_data:
                    return error_response(DATA_NOT_FOUND)

                open_ended_column = spreadsheet_service.detect_open_ended_column(
                    spreadsheet_data
                )
            print(f"Open-ended text column: {str(open_ended_column)}")

            # If high accuracy is required but detection failed, ask for a Google Form link
            if accuracy_required and not open_ended_column:
                return error_response(ACCURACY_REQUIRED)

            # Convert to DataFrame
            df = spreadsheet_service.convert_to_dataframe(
                spreadsheet_data, open_ended_column
            )

            # Prepare data using the correct column
            prepared_df, data_quality_metrics = (
                data_prep_service.prepare_column_for_clustering(df, open_ended_column)
            )

            # Convert the column to a list before passing to clustering service
            clusters = clustering_service.cluster_sentences(
                prepared_df[open_ended_column].tolist(), data_quality_metrics
            )

            print(f"Spreadsheet link: {str(spreadsheet_link)}")
            print(f"Detected Open-Ended Column: {str(open_ended_column)}")
            # print(f"Defined clusters: {str(clusters)}")

            return success_response(clusters)

        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")
