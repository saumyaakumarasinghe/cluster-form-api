import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from middleware.response_handler_middleware import error_response, success_response
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from constants.response_constants import EMPTY_FEEDBACK_LIST


class KClusteringService:
    @staticmethod
    def generate_cluster_summary(feedback_list, labels, optimal_k):
        """
        Generate a comprehensive summary of cluster contents

        Args:
            feedback_list (list): Original feedback entries
            labels (array): Cluster labels for each feedback
            optimal_k (int): Number of clusters

        Returns:
            dict: Detailed cluster summary
        """
        # Prepare detailed cluster analysis
        cluster_summary = {}

        for cluster in range(optimal_k):
            # Find feedbacks in this cluster
            cluster_feedbacks = [
                feedback
                for feedback, label in zip(feedback_list, labels)
                if label == cluster
            ]

            # Get sample size dynamically based on cluster size
            sample_size = min(
                len(cluster_feedbacks), max(5, int(len(cluster_feedbacks) * 0.1))
            )

            # Analyze cluster characteristics
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
        Create comprehensive visualizations of clustering results

        Args:
            feedback_list (list): Original feedback entries
            labels (array): Cluster labels for each feedback
            optimal_k (int): Number of clusters
            output_format (str): 'file' to save to disk or 'base64' to return as string
            output_path (str, optional): Path to save the visualization if output_format is 'file'

        Returns:
            str: Either the file path or base64 encoded string of the image
        """
        import io
        import base64
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()  # Set seaborn theme for the plot

        # Create a smaller figure with reduced size
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # Reduced the figsize
        fig.suptitle(
            "Clustering Analysis Visualization", fontsize=12, fontweight="bold"
        )

        # 1. Pie chart of cluster distribution (Left side)
        cluster_sizes = [sum(labels == i) for i in range(optimal_k)]
        axes[0, 0].pie(
            cluster_sizes,
            labels=[f"Cluster {i}" for i in range(optimal_k)],
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("pastel", optimal_k),
        )
        axes[0, 0].set_title("Cluster Distribution")
        axes[0, 0].axis("off")  # Hide axes for the pie chart

        # 2. Cluster Size Distribution (Right side)
        axes[0, 1].bar(
            range(optimal_k), cluster_sizes, color="skyblue", edgecolor="navy"
        )
        axes[0, 1].set_title("Cluster Size Distribution")
        axes[0, 1].set_xlabel("Cluster Number")
        axes[0, 1].set_ylabel("Number of Entries")

        # 3. Brief text info panel (Center of bottom row)
        axes[1, 0].axis("off")  # Hide axes for the info panel
        axes[1, 1].axis("off")  # Hide axes for the info panel
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
        )

        plt.tight_layout()

        # Handle different output formats
        if output_format == "base64":
            # Save to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Convert to base64 string
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
            return img_str

        else:  # file output
            plt.savefig(output_path, format="png", dpi=100)
            plt.close()
            return output_path

    @staticmethod
    def advanced_clustering(
        feedback_list, max_features=None, max_clusters=None, random_state=None
    ):
        """
        Perform advanced text clustering with comprehensive analysis

        Args:
            feedback_list (list): List of feedback strings
            max_features (int, optional): Maximum number of features for vectorization. If None, calculated dynamically.
            max_clusters (int, optional): Maximum number of clusters to consider. If None, calculated dynamically.
            random_state (int, optional): Random seed for reproducibility. If None, use default.

        Returns:
            dict: Detailed clustering results
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

        # Advanced Vectorization with Dynamic Configuration
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True,
            smooth_idf=True,
        )

        try:
            # Transform feedback to numerical vectors
            X = vectorizer.fit_transform(feedback_list)
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

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_array)

            # Dimensionality Reduction with dynamic number of components
            n_components = min(10, X_scaled.shape[1], X_scaled.shape[0] - 1)
            if n_components < 2:
                n_components = 2

            pca = PCA(n_components=n_components)
            X_reduced = pca.fit_transform(X_scaled)

            # Find optimal number of clusters
            def find_optimal_clusters(X, max_clusters):
                """
                Determine optimal number of clusters using silhouette score

                Args:
                    X (numpy.ndarray): Reduced feature matrix
                    max_clusters (int): Maximum clusters to evaluate

                Returns:
                    int: Optimal number of clusters
                """
                # Ensure max_clusters is valid
                max_clusters = min(max_clusters, X.shape[0] - 1, 20)
                max_clusters = max(2, max_clusters)

                # If very few samples, use minimum clusters
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

                # Default to a reasonable number if all scores are the same
                if len(set(silhouette_scores)) == 1:
                    return min(5, max_clusters)

                optimal_k = 2 + np.argmax(silhouette_scores)
                return optimal_k

            # Determine optimal clusters
            optimal_k = find_optimal_clusters(X_reduced, max_clusters)

            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=random_state,
                n_init="auto",
                init="k-means++",
            )
            labels = kmeans.fit_predict(X_reduced)

            # Compute clustering metrics safely
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

            # Generate comprehensive cluster summary
            cluster_summary = KClusteringService.generate_cluster_summary(
                feedback_list, labels, optimal_k
            )

            # Visualize clustering results
            try:
                KClusteringService.visualize_clustering_results(
                    feedback_list, labels, optimal_k
                )
            except Exception as e:
                print(f"Visualization error: {e}")

            # Detailed console output
            print("\n" + "=" * 50)

            print("\n📊 CLUSTERING METRICS:")
            print(f"• Optimal Number of Clusters: {optimal_k}")
            print(f"• Average Silhouette Score: {silhouette_avg:.4f}")
            print(f"• Calinski-Harabasz Score: {calinski_score:.4f}")
            print(f"• Davies-Bouldin Score: {davies_score:.4f}")

            print("\n🧩 CLUSTER BREAKDOWN:")
            for cluster, details in cluster_summary.items():
                print(f"\n✦ Cluster {cluster}:")
                print(f"  • Total Entries: {details['size']}")
                print("  • Sample Feedbacks: ")
                for i, feedback in enumerate(
                    details["feedbacks"][: details["sample_size"]], 1
                ):
                    print(f"    {i}. {feedback}")

            # Prepare feature importance for each cluster if needed
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


# The function would then be called in your route handler
def advanced_clustering_handler(
    feedback_list, max_features=None, max_clusters=None, random_state=None
):
    results = KClusteringService.advanced_clustering(
        feedback_list,
        max_features=max_features,
        max_clusters=max_clusters,
        random_state=random_state,
    )

    # Generate visualization and add to results
    try:
        # For base64 (can be directly embedded in HTML)
        visualization = KClusteringService.visualize_clustering_results(
            feedback_list,
            results["labels"],
            results["optimal_clusters"],
            output_format="base64",
        )
        results["visualization_base64"] = f"data:image/png;base64,{visualization}"

    except Exception as e:
        results["visualization_error"] = str(e)

    return results
