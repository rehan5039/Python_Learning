"""
Hierarchical Clustering Implementation
===================================

This module provides implementations of agglomerative and divisive hierarchical clustering
algorithms with various linkage criteria and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AgglomerativeClustering:
    """
    Agglomerative Hierarchical Clustering implementation.
    
    Parameters:
    -----------
    n_clusters : int, default=2
        Number of clusters to find
    linkage : str, default='ward'
        Linkage criterion ('ward', 'complete', 'average', 'single')
    metric : str, default='euclidean'
        Distance metric to use
    
    Attributes:
    -----------
    labels : array, shape (n_samples,)
        Cluster labels for each point
    linkage_matrix : array
        Linkage matrix for dendrogram visualization
    """
    
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
    
    def fit(self, X):
        """
        Fit the hierarchical clustering model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster
        """
        # Compute linkage matrix
        if self.linkage == 'ward':
            # Ward linkage only works with euclidean distance
            self.linkage_matrix = linkage(X, method='ward')
        else:
            # For other linkages, we can use any distance metric
            distances = pdist(X, metric=self.metric)
            self.linkage_matrix = linkage(distances, method=self.linkage)
        
        # Form flat clusters
        self.labels = fcluster(self.linkage_matrix, self.n_clusters, criterion='maxclust')
        return self
    
    def fit_predict(self, X):
        """
        Fit the hierarchical clustering model and return cluster labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        return self.fit(X).labels


def plot_dendrogram(linkage_matrix, labels=None, truncate_mode=None, p=30, 
                   color_threshold=None, figsize=(12, 8)):
    """
    Plot dendrogram for hierarchical clustering.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from scipy.cluster.hierarchy.linkage
    labels : array-like, optional
        Labels for leaf nodes
    truncate_mode : str, optional
        Truncation mode ('level', 'lastp', None)
    p : int, default=30
        Parameter for truncation
    color_threshold : float, optional
        Threshold for coloring clusters
    figsize : tuple, default=(12, 8)
        Figure size
    """
    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=labels,
        truncate_mode=truncate_mode,
        p=p,
        color_threshold=color_threshold,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()


def compare_linkage_methods(X, n_clusters=4):
    """
    Compare different linkage methods for hierarchical clustering.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    n_clusters : int, default=4
        Number of clusters to form
        
    Returns:
    --------
    results : dict
        Dictionary containing results for each linkage method
    """
    linkage_methods = ['ward', 'complete', 'average', 'single']
    results = {}
    
    for method in linkage_methods:
        try:
            if method == 'ward':
                # Ward only works with euclidean distance
                clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
            else:
                clustering = AgglomerativeClustering(n_clusters=n_clusters, 
                                                   linkage=method, 
                                                   metric='euclidean')
            
            labels = clustering.fit_predict(X)
            
            # Calculate metrics
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = 0
            
            results[method] = {
                'labels': labels,
                'linkage_matrix': clustering.linkage_matrix,
                'silhouette_score': silhouette
            }
        except Exception as e:
            print(f"Error with {method} linkage: {e}")
            results[method] = None
    
    return results


def cut_dendrogram_at_height(linkage_matrix, height):
    """
    Cut dendrogram at a specific height to form clusters.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from scipy.cluster.hierarchy.linkage
    height : float
        Height at which to cut the dendrogram
        
    Returns:
    --------
    labels : array
        Cluster labels
    """
    return fcluster(linkage_matrix, height, criterion='distance')


def find_optimal_height(linkage_matrix, max_clusters=10):
    """
    Find optimal cutting height for dendrogram based on desired number of clusters.
    
    Parameters:
    -----------
    linkage_matrix : array
        Linkage matrix from scipy.cluster.hierarchy.linkage
    max_clusters : int, default=10
        Maximum number of clusters to consider
        
    Returns:
    --------
    optimal_height : float
        Optimal height for cutting dendrogram
    """
    # Get unique heights from linkage matrix
    heights = np.unique(linkage_matrix[:, 2])
    heights = np.sort(heights)[::-1]  # Sort in descending order
    
    # Return height that gives max_clusters
    if len(heights) >= max_clusters:
        return heights[max_clusters - 1]
    else:
        return heights[-1]


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=150, centers=4, cluster_std=1.0, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply agglomerative clustering
    clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = clustering.fit_predict(X_scaled)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.title('Agglomerative Clustering Results')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.grid(True, alpha=0.3)
    
    # Dendrogram
    plt.subplot(1, 2, 2)
    plot_dendrogram(clustering.linkage_matrix, truncate_mode='level', p=5)
    
    plt.tight_layout()
    plt.show()
    
    # Compare linkage methods
    print("Comparing linkage methods:")
    comparison_results = compare_linkage_methods(X_scaled, n_clusters=4)
    
    for method, result in comparison_results.items():
        if result is not None:
            print(f"{method.capitalize()} linkage:")
            print(f"  Silhouette Score: {result['silhouette_score']:.4f}")
    
    # Demonstrate cutting at different heights
    print("\nDemonstrating cutting at different heights:")
    heights_to_test = [5, 10, 15]
    for height in heights_to_test:
        cut_labels = cut_dendrogram_at_height(clustering.linkage_matrix, height)
        unique_clusters = len(np.unique(cut_labels))
        print(f"Cut at height {height}: {unique_clusters} clusters")
        
        if unique_clusters > 1:
            silhouette = silhouette_score(X_scaled, cut_labels)
            print(f"  Silhouette Score: {silhouette:.4f}")