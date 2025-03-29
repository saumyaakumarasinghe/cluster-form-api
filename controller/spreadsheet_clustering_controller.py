from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND
from constants.link_constants import SPREADSHEET_RANGE
from services.spreadsheet_services import SpreadsheetService
# from services.clustering_services import ClusteringService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService


def cluster_spreadsheet(
    link: str,
    spreadsheet_service: SpreadsheetService,
    # clustering_service: ClusteringService,
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
            # clusters = clustering_service.cluster_sentences(
            #     prepared_df[target_column].tolist(), data_quality_metrics
            # )
            clusters = kclustering_service.advanced_clustering(
                prepared_df[target_column].tolist()
            )

            print(f"Spreadsheet link: {str(link)}")
            print(f"Spreadsheet ID: {str(spreadsheet_id)}")
            print(f"Spreadsheet column: {str(SPREADSHEET_RANGE)}")
            print(f"Defined clusters---------------------: {str(clusters)}")

            return success_response("clusters")

        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")
