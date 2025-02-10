from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def convert_numpy_to_native(data):
    """Recursively convert numpy types (int64, float64) to native Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()  # convert numpy array to list
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)  # convert numpy int to native int
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)  # convert numpy float to native float
    elif isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    return data  # return the data as is if it's not a numpy type


class ClusteringService:
    def __init__(self):
        self.scaler = StandardScaler()

    def _preprocess_data(self, data: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess list data for clustering."""
        series_data = pd.Series(data)
        valid_mask = series_data.notna() & (series_data != "")
        print(f"Valid mask count: {valid_mask.sum()}")

        # convert text data to numerical features using TF-IDF
        tfidf = TfidfVectorizer()
        text_data = series_data[valid_mask].values
        tfidf_matrix = tfidf.fit_transform(text_data)

        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        clean_data = tfidf_matrix.toarray()  # convert sparse matrix to dense array
        print(f"Clean data count: {len(clean_data)}")

        if len(clean_data) >= 2:
            clean_data = self.scaler.fit_transform(clean_data)
        return clean_data, valid_mask

    def _estimate_dbscan_params(self, data: np.ndarray) -> Dict[str, float]:
        """Estimate optimal DBSCAN parameters."""
        n_neighbors = min(len(data) - 1, 5)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
        distances, _ = nbrs.kneighbors(data)
        eps = np.mean(distances[:, -1]) * 1.5
        if len(data) <= 20:
            min_samples = 2
        elif len(data) <= 50:
            min_samples = 3
        else:
            min_samples = 4
        return {"eps": eps, "min_samples": min_samples}

    def _get_data_quality_metrics(self, data: List[Any]) -> Dict[str, int]:
        """Calculate data quality metrics from list data."""
        series_data = pd.Series(data)
        return {
            "total_records": len(series_data),
            "null_count": series_data.isna().sum(),
            "empty_string_count": series_data.astype(str).str.strip().eq("").sum(),
        }

    def cluster_sentences(self, data: List[Any]) -> Dict[str, Any]:
        """Perform clustering on list data with proper JSON serialization."""
        try:
            # get data quality metrics
            quality_metrics = self._get_data_quality_metrics(data)

            # preprocess data
            clean_data, valid_mask = self._preprocess_data(data)

            if len(clean_data) < 2:
                return convert_numpy_to_native(
                    {
                        "error": "Insufficient valid data for clustering",
                        "data_quality": quality_metrics,
                        "status": "failed",
                    }
                )

            # estimate DBSCAN parameters
            params = self._estimate_dbscan_params(clean_data)

            # perform clustering
            dbscan = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
            clusters = dbscan.fit_predict(clean_data)

            # create a list of cluster assignments with -1 for invalid data
            all_clusters = [-1] * len(data)  # initialize all as noise points
            valid_indices = np.where(valid_mask)[0]
            for idx, cluster in zip(valid_indices, clusters):
                all_clusters[idx] = int(cluster)  # convert numpy.int64 to Python int

            # group sentences by cluster label
            cluster_sentences = {}
            for idx, cluster in zip(valid_indices, clusters):
                if cluster != -1:  # ignore noise points
                    if cluster not in cluster_sentences:
                        cluster_sentences[cluster] = []
                    cluster_sentences[cluster].append(data[idx])

            # convert numpy.int64 keys to Python int for JSON serialization
            cluster_sentences = {int(k): v for k, v in cluster_sentences.items()}

            # calculate cluster statistics
            cluster_stats = {}
            for cluster_id in set(clusters):
                if cluster_id != -1:
                    mask = clusters == cluster_id
                    cluster_stats[f"cluster_{cluster_id}"] = {
                        "size": int(np.sum(mask)),
                        "mean": float(np.mean(clean_data[mask])),
                        "std": (
                            float(np.std(clean_data[mask])) if len(mask) > 1 else 0.0
                        ),
                        "min": float(np.min(clean_data[mask])),
                        "max": float(np.max(clean_data[mask])),
                    }

            # convert all numpy numbers to Python native types
            clustering_results = {
                "clusters": convert_numpy_to_native(
                    all_clusters
                ),  # convert all cluster IDs to Python int
                "cluster_sentences": convert_numpy_to_native(
                    cluster_sentences
                ),  # add sentences by cluster
                "cluster_stats": convert_numpy_to_native(cluster_stats),
                "data_quality": convert_numpy_to_native(quality_metrics),
                "parameters_used": convert_numpy_to_native(params),
                "status": "success",
                "metadata": convert_numpy_to_native(
                    {
                        "original_rows": int(quality_metrics["total_records"]),
                        "processed_rows": int(len(clean_data)),
                        "noise_points": int(sum(1 for c in all_clusters if c == -1)),
                    }
                ),
            }

            return convert_numpy_to_native(clustering_results)

        except Exception as e:
            return convert_numpy_to_native({"error": str(e), "status": "failed"})
