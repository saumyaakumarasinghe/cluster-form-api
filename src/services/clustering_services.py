from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from typing import List, Dict
import logging


def _find_optimal_eps(X: np.ndarray) -> float:
    """
    Find optimal eps parameter for DBSCAN using distance distribution.
    For small datasets, we use a more lenient approach.
    """
    # Convert to dense array for distance computation
    X_dense = X.toarray()

    # Compute pairwise distances
    from sklearn.metrics.pairwise import cosine_distances

    distances = cosine_distances(X_dense)

    # Get the mean of the distances to the k nearest neighbors
    k = min(3, len(distances) - 1)  # Adjust k based on dataset size
    distances.sort(axis=1)
    knn_distances = distances[:, 1 : k + 1]

    # Use a percentile of the mean distances as eps
    eps = np.percentile(knn_distances, 75)  # Use 75th percentile

    # Ensure eps is within reasonable bounds
    eps = min(max(eps, 0.3), 0.8)

    return float(eps)


def _get_cluster_keywords(
    X: np.ndarray, labels: np.ndarray, vectorizer: TfidfVectorizer, n_keywords: int = 3
) -> Dict[int, List[str]]:
    """Extract top keywords for each cluster."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_keywords = {}

    unique_labels = set(int(label) for label in labels)  # Convert numpy.int64 to int

    for label in unique_labels:
        cluster_docs = X[labels == label]
        centroid = cluster_docs.mean(axis=0).A1
        top_indices = centroid.argsort()[-n_keywords:][::-1]
        cluster_keywords[int(label)] = feature_names[
            top_indices
        ].tolist()  # Convert label to int

    return cluster_keywords


def cluster_sentences(sentences: List[str]) -> Dict:
    """
    Cluster sentences and return clusters with their keywords.
    Optimized for spreadsheet data processing.
    """
    if not sentences:
        return {"error": "No sentences provided"}

    try:
        # Process sentences to ensure they're all strings
        processed_sentences = [str(s).strip() for s in sentences if str(s).strip()]

        if not processed_sentences:
            return {"error": "No valid sentences after preprocessing"}

        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )

        # Convert sentences to TF-IDF features
        X = vectorizer.fit_transform(processed_sentences)

        # Adjust parameters based on dataset size
        min_samples = 1 if len(processed_sentences) < 10 else 2
        eps = 0.5  # Fixed eps for more consistent clustering

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(X)

        # Organize sentences into clusters
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(processed_sentences[i])

        # Extract keywords for each cluster
        feature_names = vectorizer.get_feature_names_out()
        keywords = {}

        for label in set(int(l) for l in labels):
            cluster_docs = X[labels == label]
            if cluster_docs.shape[0] > 0:
                centroid = cluster_docs.mean(axis=0).A1
                top_indices = centroid.argsort()[-3:][::-1]  # Get top 3 keywords
                keywords[int(label)] = [feature_names[i] for i in top_indices]

        # Calculate statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        stats = {
            "total_sentences": len(processed_sentences),
            "num_clusters": n_clusters,
            "noise_points": n_noise,
            "eps_value": float(eps),
            "average_cluster_size": (
                float((len(processed_sentences) - n_noise) / max(n_clusters, 1))
                if n_clusters > 0
                else 0.0
            ),
        }

        return {"clusters": clusters, "keywords": keywords, "stats": stats}

    except Exception as e:
        logging.error(f"Clustering error: {str(e)}")
        return {"error": f"Clustering failed: {str(e)}"}

    # def cluster_sentences(sentences: List[str]) -> Dict:
    """
    Cluster sentences and return clusters with their keywords.
    Optimized for both small and large datasets.
    """
    if not sentences:
        return {"error": "No sentences provided"}

    try:
        # Initialize vectorizer with adjusted parameters
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,  # Allow terms that appear in single documents
        )

        # Convert sentences to TF-IDF features
        X = vectorizer.fit_transform(sentences)

        # Adjust parameters based on dataset size
        min_samples = 1 if len(sentences) < 10 else 2

        # Find optimal eps
        eps = _find_optimal_eps(X)

        # Apply DBSCAN clustering with adjusted parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(X)

        # Organize sentences into clusters
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)  # Convert numpy.int64 to regular Python int
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])

        # Get keywords for each cluster
        keywords = _get_cluster_keywords(X, labels, vectorizer)

        # Calculate statistics
        unique_labels = set(
            int(label) for label in labels
        )  # Convert to regular Python int
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        stats = {
            "total_sentences": len(sentences),
            "num_clusters": n_clusters,
            "noise_points": n_noise,
            "eps_value": float(eps),  # Convert numpy.float64 to regular Python float
            "average_cluster_size": (
                float((len(sentences) - n_noise) / max(n_clusters, 1))
                if n_clusters > 0
                else 0.0
            ),
        }

        return {"clusters": clusters, "keywords": keywords, "stats": stats}

    except Exception as e:
        logging.error(f"Clustering error: {str(e)}")
        return {"error": f"Clustering failed: {str(e)}"}
