from middleware.response_handler_middleware import error_response, success_response
from constants.response_constants import EMPTY_LINK, INVALID_LINK, DATA_NOT_FOUND
from constants.link_constants import SPREADSHEET_RANGE
from services.spreadsheet_services import SpreadsheetService
# from services.clustering_services import ClusteringService
from services.kmeans_service import KClusteringService
from services.data_preparation_service import DataPreparationService
import numpy as np

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
            print(f"DATA-------: {prepared_df[target_column].tolist()}")
            print(f"DATA-------: {data_quality_metrics}")
            clustering_results = kclustering_service.advanced_clustering(
                prepared_df[target_column].tolist()
            )

            print(f"Spreadsheet link: {str(link)}")
            print(f"Spreadsheet ID: {str(spreadsheet_id)}")
            print(f"Spreadsheet column: {str(SPREADSHEET_RANGE)}")

            # Convert any numpy.int64 to native Python int for JSON serialization
            optimal_clusters = int(clustering_results['optimal_clusters']) if isinstance(clustering_results['optimal_clusters'], np.int64) else clustering_results['optimal_clusters']
            silhouette_score = float(clustering_results['silhouette_score']) if isinstance(clustering_results['silhouette_score'], np.int64) else clustering_results['silhouette_score']
            calinski_score = float(clustering_results['calinski_score']) if isinstance(clustering_results['calinski_score'], np.int64) else clustering_results['calinski_score']
            davies_score = float(clustering_results['davies_score']) if isinstance(clustering_results['davies_score'], np.int64) else clustering_results['davies_score']

            # If cluster_summary has numpy.int64 values, convert them as well
            cluster_summary = {key: [int(item) if isinstance(item, np.int64) else item for item in value] for key, value in clustering_results['cluster_summary'].items()}

            # Send the clusters in the response
            return success_response({
                "message": "Clustering operation completed successfully.",
                "optimal_clusters": optimal_clusters,
                "silhouette_score": silhouette_score,
                "calinski_score": calinski_score,
                "davies_score": davies_score,
                "cluster_summary": cluster_summary,
            })


        except Exception as e:
            return error_response(f"Error accessing spreadsheet data: {str(e)}")

    except Exception as e:
        return error_response(f"Error in clustering process: {str(e)}")
