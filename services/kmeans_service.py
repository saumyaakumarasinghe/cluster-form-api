import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from wordcloud import WordCloud

import seaborn as sns

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
                feedback for feedback, label in zip(feedback_list, labels) 
                if label == cluster
            ]
            
            # Analyze cluster characteristics
            cluster_summary[cluster] = {
                'size': len(cluster_feedbacks),
                'feedbacks': cluster_feedbacks,
                'sample_size': min(5, len(cluster_feedbacks))
            }
        
        return cluster_summary

    @staticmethod
    def visualize_clustering_results(feedback_list, labels, optimal_k):
        """
        Create comprehensive visualizations of clustering results
        
        Args:
            feedback_list (list): Original feedback entries
            labels (array): Cluster labels for each feedback
            optimal_k (int): Number of clusters
        """
        sns.set_theme()  # Set seaborn theme for the plot

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Analysis Visualization', fontsize=16, fontweight='bold')

        # 1. Cluster Size Distribution
        cluster_sizes = [sum(labels == i) for i in range(optimal_k)]
        axes[0, 0].bar(range(optimal_k), cluster_sizes, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster Number')
        axes[0, 0].set_ylabel('Number of Feedback Entries')

        # 2. Word Cloud for Each Cluster (Simplified)
        def create_wordcloud(texts):
            """Generate a word cloud from a list of texts"""
            text = ' '.join(texts)
            wordcloud = WordCloud(
                width=400, 
                height=200, 
                background_color='white'
            ).generate(text)
            return wordcloud
        
        # Word Clouds for first two clusters
        cluster_summary = KClusteringService.generate_cluster_summary(feedback_list, labels, optimal_k)

        for i in range(min(2, optimal_k)):
            cluster_texts = cluster_summary[i]['feedbacks']
            wordcloud = create_wordcloud(cluster_texts)

            axes[0, 1].imshow(wordcloud, interpolation='bilinear')
            axes[0, 1].set_title(f'Word Cloud for Cluster {i}')
            axes[0, 1].axis('off')

        # 3. Feedback Length Distribution per Cluster
        feedback_lengths = [len(feedback.split()) for feedback in feedback_list]
        axes[1, 0].boxplot(
            [
                [length for length, label in zip(feedback_lengths, labels) if label == cluster] 
                for cluster in range(optimal_k)
            ]
        )
        axes[1, 0].set_title('Feedback Length Distribution per Cluster')
        axes[1, 0].set_xlabel('Cluster Number')
        axes[1, 0].set_ylabel('Number of Words')

        # 4. Sentiment Distribution (Simple Heuristic)
        def simple_sentiment_score(text):
            """Basic sentiment scoring"""
            positive_words = ['good', 'great', 'awesome', 'delicious', 'yummy', 'best']
            negative_words = ['bad', 'terrible', 'worst', 'horrible']

            text_lower = text.lower()
            positive_count = sum(word in text_lower for word in positive_words)
            negative_count = sum(word in text_lower for word in negative_words)

            return positive_count - negative_count

        sentiment_scores = [simple_sentiment_score(feedback) for feedback in feedback_list]

        axes[1, 1].boxplot(
            [
                [score for score, label in zip(sentiment_scores, labels) if label == cluster] 
                for cluster in range(optimal_k)
            ]
        )
        axes[1, 1].set_title('Sentiment Distribution per Cluster')
        axes[1, 1].set_xlabel('Cluster Number')
        axes[1, 1].set_ylabel('Sentiment Score')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def advanced_clustering(self, feedback_list):
        """
        Perform advanced text clustering with comprehensive analysis
        
        Args:
            feedback_list (list): List of feedback strings
        
        Returns:
            dict: Detailed clustering results
        """
        # Advanced Vectorization with Detailed Configuration
        vectorizer = TfidfVectorizer(
            stop_words='english',  # Remove common English stop words
            lowercase=True,  # Convert to lowercase
            ngram_range=(1, 3),  # Capture unigrams, bigrams, and trigrams
            max_features=500,  # Limit to top 500 features
            sublinear_tf=True,  # Apply sublinear scaling to term frequencies
            smooth_idf=True  # Add small value to IDF to prevent division by zero
        )

        # Transform feedback to numerical vectors
        X = vectorizer.fit_transform(feedback_list)
        X_array = X.toarray()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array)

        # Dimensionality Reduction
        pca = PCA(n_components=min(5, X_scaled.shape[1]))
        X_reduced = pca.fit_transform(X_scaled)

        # Find optimal number of clusters
        def find_optimal_clusters(X, max_clusters=10):
            """
            Determine optimal number of clusters using multiple metrics
            
            Args:
                X (numpy.ndarray): Reduced feature matrix
                max_clusters (int): Maximum clusters to evaluate
            
            Returns:
                int: Optimal number of clusters
            """
            max_clusters = min(max_clusters, len(X) - 1)

            silhouette_scores = []
            calinski_scores = []
            davies_scores = []

            for k in range(2, max_clusters + 1):
                kmeans = KMeans(
                    n_clusters=k, 
                    random_state=42, 
                    n_init=20,  # Multiple initializations
                    init='k-means++'
                )
                labels = kmeans.fit_predict(X)

                silhouette_scores.append(silhouette_score(X, labels))
                calinski_scores.append(calinski_harabasz_score(X, labels))
                davies_scores.append(davies_bouldin_score(X, labels))

            optimal_k = 2 + np.argmax([
                0.4 * silhouette_scores[i] + 
                0.3 * calinski_scores[i] - 
                0.3 * davies_scores[i] 
                for i in range(len(silhouette_scores))
            ])

            return optimal_k

        # Determine optimal clusters
        optimal_k = find_optimal_clusters(X_reduced)

        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=optimal_k, 
            random_state=42, 
            n_init=20,
            init='k-means++',
            algorithm='lloyd'
        )
        labels = kmeans.fit_predict(X_reduced)

        # Compute clustering metrics
        silhouette_avg = silhouette_score(X_reduced, labels)
        calinski_score = calinski_harabasz_score(X_reduced, labels)
        davies_score = davies_bouldin_score(X_reduced, labels)

        # Generate comprehensive cluster summary
        cluster_summary = KClusteringService.generate_cluster_summary(feedback_list, labels, optimal_k)

        # Visualize clustering results
        KClusteringService.visualize_clustering_results(feedback_list, labels, optimal_k)

        # Detailed console output
        print("\n" + "=" * 50)
        print("🔍 COMPREHENSIVE CLUSTERING ANALYSIS 🔍".center(50))
        print("=" * 50)

        print("\n📊 CLUSTERING METRICS:")
        print(f"• Optimal Number of Clusters: {optimal_k}")
        print(f"• Average Silhouette Score: {silhouette_avg:.4f}")
        print(f"• Calinski-Harabasz Score: {calinski_score:.4f}")
        print(f"• Davies-Bouldin Score: {davies_score:.4f}")

        print("\n🧩 CLUSTER BREAKDOWN:")
        for cluster, details in cluster_summary.items():
            print(f"\n✦ Cluster {cluster}:")
            print(f"  • Total Entries: {details['size']}")
            print("  • Sample Feedbacks:")
            for i, feedback in enumerate(details['feedbacks'][:details['sample_size']], 1):
                print(f"    {i}. {feedback}")

        return {
            'optimal_clusters': optimal_k,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score,
            'davies_score': davies_score,
            'cluster_summary': cluster_summary,
            'labels': labels
        }

# # Example usage
# restaurant_feedback = [
#     "It was delicious", 
#     "bad", 
#     "VERY bad but i ate it", 
#     "Pretty good", 
#     "woow yummy", 
#     "It was good I ate a salad", 
#     "Maybe good but not best", 
#     "I ate a mix rice for today lunch. It was ok", 
#     "Pasta made my life better with cheese and mayo"
# ]

# # Run advanced clustering
# results = KClusteringService.advanced_clustering(restaurant_feedback)
