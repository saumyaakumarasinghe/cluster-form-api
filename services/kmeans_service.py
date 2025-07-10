import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from middleware.response_handler_middleware import error_response
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants.response_constants import EMPTY_FEEDBACK_LIST


class KClusteringService:
    """
    Service for performing advanced text clustering using KMeans.
    Handles vectorization, dimensionality reduction, cluster analysis, and visualization.
    """
    @staticmethod
    def generate_cluster_summary(feedback_list, labels, optimal_k):
        """
        Generate a summary of the contents of each cluster.
        Args:
            feedback_list (list): List of original feedback entries.
            labels (array): Cluster labels for each feedback.
            optimal_k (int): Number of clusters.
        Returns:
            dict: Summary with size, feedbacks, and sample size for each cluster.
        """
        cluster_summary = {}
        for cluster in range(optimal_k):
            # Get all feedbacks assigned to this cluster
            cluster_feedbacks = [
                feedback
                for feedback, label in zip(feedback_list, labels)
                if label == cluster
            ]
            # Use a sample size for previewing cluster contents
            sample_size = min(
                len(cluster_feedbacks), max(5, int(len(cluster_feedbacks) * 0.1))
            )
            cluster_summary[cluster] = {
                "size": len(cluster_feedbacks),
                "feedbacks": cluster_feedbacks,
                "sample_size": sample_size,
            }
        return cluster_summary

    @staticmethod
    def visualize_clustering_results(
        feedback_list, labels, optimal_k, output_format="base64", output_path=None
    ):
        """
        Create a visualization of clustering results, including a pie chart, bar chart, and summary info.
        Args:
            feedback_list (list): Original feedback entries.
            labels (array): Cluster labels for each feedback.
            optimal_k (int): Number of clusters.
            output_format (str): 'base64' to return as string, 'file' to save to disk.
            output_path (str, optional): Path to save the visualization if output_format is 'file'.
        Returns:
            str: Base64-encoded image or file path.
        """
        import io
        import base64
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()  # Use a clean theme for plots
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of plots
        fig.suptitle(
            "Clustering Analysis Visualization", fontsize=24, fontweight="bold"
        )
        cluster_sizes = [sum(labels == i) for i in range(optimal_k)]  # Count per cluster
        # Pie chart: distribution of entries across clusters
        axes[0, 0].pie(
            cluster_sizes,
            labels=[f"Cluster {i}" for i in range(optimal_k)],
            autopct="%1.1%%",
            startangle=90,
            colors=sns.color_palette("pastel", optimal_k),
            textprops={"fontweight": "bold"},
        )
        axes[0, 0].set_title("Cluster Distribution", fontsize=20, fontweight="bold")
        axes[0, 0].axis("off")
        # Bar chart: size of each cluster
        axes[0, 1].bar(
            range(optimal_k),
            cluster_sizes,
            color="skyblue",
            edgecolor="navy",
        )
        axes[0, 1].set_title(
            "Cluster Size Distribution", fontsize=20, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Cluster Number", fontsize=15, fontweight="bold")
        axes[0, 1].set_ylabel("Number of Entries", fontsize=15, fontweight="bold")
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        # Info panel: summary statistics
        info_text = (
            f"Total Entries: {len(feedback_list)}\n"
            f"Number of Clusters: {optimal_k}\n"
            f"Largest Cluster: Cluster {max(range(optimal_k), key=lambda i: cluster_sizes[i])} "
            f"({max(cluster_sizes)} entries, "
            f"{(max(cluster_sizes)/len(feedback_list)*100):.1f}%)\n"
            f"Smallest Cluster: Cluster {min(range(optimal_k), key=lambda i: cluster_sizes[i])} "
            f"({min(cluster_sizes)} entries)"
        )
        axes[1, 0].text(
            0.5,
            0.5,
            info_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[1, 0].transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        if output_format == "base64":
            buf = io.BytesIO()  # Save to buffer
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")  # Encode as base64
            plt.close()
            return img_str
        else:
            plt.savefig(output_path, format="png", dpi=100)
            plt.close()
            return output_path

    @staticmethod
    def advanced_clustering(
        feedback_list, max_features=None, max_clusters=None, random_state=None
    ):
        """
        Perform advanced text clustering using KMeans, with dynamic parameter selection and metrics.
        Steps:
            1. Vectorize text using TF-IDF.
            2. Standardize features.
            3. Reduce dimensionality with PCA.
            4. Find optimal number of clusters using silhouette score.
            5. Run KMeans clustering.
            6. Compute cluster metrics and feature importance.
        Args:
            feedback_list (list): List of feedback strings.
            max_features (int, optional): Max features for vectorization.
            max_clusters (int, optional): Max clusters to consider.
            random_state (int, optional): Random seed.
        Returns:
            dict: Clustering results, metrics, and summaries.
        """
        if not feedback_list:
            return error_response(EMPTY_FEEDBACK_LIST)

        # Dynamic parameters based on dataset size
        n_samples = len(feedback_list)

        # Adjust max_features based on data size if not provided
        if max_features is None:
            max_features = min(5000, max(100, n_samples * 5))

        # Adjust max_clusters based on data size if not provided
        if max_clusters is None:
            max_clusters = min(20, max(2, n_samples // 10))

        # Set random_state for reproducibility if not provided
        if random_state is None:
            random_state = 42
        # Step 1: Vectorize text data
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True,
            smooth_idf=True,
        )

        try:
            X = vectorizer.fit_transform(feedback_list)  # Text to vectors
            X_array = X.toarray()

            # Handle case with too few samples or features
            if X_array.shape[0] < 2 or X_array.shape[1] < 1:
                return {
                    "error": "Not enough data for meaningful clustering",
                    "optimal_clusters": 1,
                    "labels": np.zeros(len(feedback_list), dtype=int),
                    "cluster_summary": KClusteringService.generate_cluster_summary(
                        feedback_list, np.zeros(len(feedback_list), dtype=int), 1
                    ),
                }
            # Step 2: Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)
            # Step 3: Dimensionality reduction with PCA
            n_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
            if n_components < 2:
                n_components = 2
            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)
            # Step 4: Find optimal number of clusters
            def find_optimal_clusters(X, max_clusters):
                max_clusters = min(max_clusters, X.shape[0] - 1, 20)
                max_clusters = max(2, max_clusters)
                if X.shape[0] <= 3:
                    return 2
                silhouette_scores = []
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=random_state,
                        n_init="auto",
                        init="k-means++",
                    )
                    labels = kmeans.fit_predict(X)
                    try:
                        sil_score = silhouette_score(X, labels)
                        silhouette_scores.append(sil_score)
                    except:
                        silhouette_scores.append(-1)
                if len(set(silhouette_scores)) == 1:
                    return min(5, max_clusters)

                optimal_k = 2 + np.argmax(silhouette_scores)
                return optimal_k

            # Determine optimal clusters
            optimal_k = find_optimal_clusters(X_reduced, max_clusters)
            # Step 5: Run KMeans clustering
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=random_state,
                n_init="auto",
                init="k-means++",
            )
            labels = kmeans.fit_predict(X_reduced)
            # Step 6: Compute clustering metrics
            try:
                silhouette_avg = silhouette_score(X_reduced, labels)
            except:
                silhouette_avg = -1

            try:
                calinski_score = calinski_harabasz_score(X_reduced, labels)
            except:
                calinski_score = 0

            try:
                davies_score = davies_bouldin_score(X_reduced, labels)
            except:
                davies_score = 1
            # Generate cluster summary
            cluster_summary = KClusteringService.generate_cluster_summary(
                feedback_list, labels, optimal_k
            )
            # Optionally generate visualization (side effect)
            try:
                KClusteringService.visualize_clustering_results(
                    feedback_list, labels, optimal_k
                )
            except Exception as e:
                print(f"Visualization error: {e}")
            # Print metrics for debugging
            print("\n" + "=" * 50)
            print(f"> Optimal Number of Clusters: {optimal_k}")
            print(f"> Average Silhouette Score: {silhouette_avg:.4f}")
            print(f"> Calinski-Harabasz Score: {calinski_score:.4f}")
            print(f"> Davies-Bouldin Score: {davies_score:.4f}")
            # Feature importance for each cluster
            feature_importance = {}
            if hasattr(vectorizer, "get_feature_names_out"):
                feature_names = vectorizer.get_feature_names_out()
                for cluster in range(optimal_k):
                    # Get cluster center
                    center = kmeans.cluster_centers_[cluster]
                    # Get top features
                    top_indices = np.argsort(center)[-10:]
                    top_features = [(feature_names[i], center[i]) for i in top_indices]
                    feature_importance[cluster] = top_features

            return {
                "optimal_clusters": optimal_k,
                "silhouette_score": silhouette_avg,
                "calinski_score": calinski_score,
                "davies_score": davies_score,
                "cluster_summary": cluster_summary,
                "labels": labels,
                "feature_importance": feature_importance,
            }

        except Exception as e:
            print(f"Clustering error: {str(e)}")
            # Fallback to a single cluster in case of error
            labels = np.zeros(len(feedback_list), dtype=int)
            return {
                "error": f"Clustering failed: {str(e)}",
                "optimal_clusters": 1,
                "labels": labels,
                "cluster_summary": KClusteringService.generate_cluster_summary(
                    feedback_list, labels, 1
                ),
            }
