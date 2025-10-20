"""
K-Means Clustering Implementation
===============================

This module provides a comprehensive implementation of the K-Means clustering algorithm
with various enhancements and evaluation techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    Parameters:
    -----------
    k : int, default=3
        Number of clusters
    max_iters : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    init_method : str, default='random'
        Initialization method ('random' or 'k-means++')
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes:
    -----------
    centroids : array, shape (k, n_features)
        Coordinates of cluster centroids
    labels : array, shape (n_samples,)
        Labels of each point
    inertia : float
        Sum of squared distances of samples to their closest centroid
    """
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, init_method='k-means++', random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
    
    def _initialize_centroids(self, X):
        """Initialize centroids using specified method."""
        n_samples, n_features = X.shape
        
        if self.init_method == 'random':
            # Random initialization
            centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        elif self.init_method == 'k-means++':
            # K-means++ initialization
            centroids = np.zeros((self.k, n_features))
            # Choose first centroid randomly
            centroids[0] = X[np.random.choice(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.k):
                # Calculate distances to nearest centroid
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in X])
                # Probability proportional to squared distance
                probabilities = distances / distances.sum()
                # Choose next centroid
                centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
        else:
            raise ValueError("init_method must be 'random' or 'k-means++'")
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid."""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """Update centroids based on current assignments."""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            if np.sum(labels == i) > 0:
                centroids[i] = X[labels == i].mean(axis=0)
            else:
                # If no points assigned, keep previous centroid
                centroids[i] = self.centroids[i]
        return centroids
    
    def fit(self, X):
        """
        Compute K-Means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iteratively update centroids
        for i in range(self.max_iters):
            # Assign points to clusters
            labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
                
            self.centroids = new_centroids
        
        self.labels = labels
        self.inertia = np.sum([np.linalg.norm(X[self.labels == i] - self.centroids[i])**2 
                              for i in range(self.k)])
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self.fit(X).labels


def find_optimal_clusters(X, k_range=range(1, 11), random_state=42):
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing inertia and silhouette scores for each k
    """
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        if k == 1:
            inertias.append(np.sum((X - X.mean(axis=0))**2))
            silhouette_scores.append(0)
        else:
            kmeans = KMeans(k=k, random_state=random_state)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia)
            silhouette_scores.append(silhouette_score(X, labels))
    
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def plot_elbow_curve(results):
    """Plot elbow curve to visualize optimal number of clusters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow curve
    ax1.plot(results['k_values'], results['inertias'], 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)
    
    # Silhouette scores
    ax2.plot(results['k_values'][1:], results['silhouette_scores'][1:], 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_initialization_methods(X, k=4, random_state=42):
    """
    Compare different initialization methods for K-Means.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    k : int, default=4
        Number of clusters
    random_state : int, default=42
        Random seed for reproducibility
    """
    methods = ['random', 'k-means++']
    results = {}
    
    for method in methods:
        kmeans = KMeans(k=k, init_method=method, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        results[method] = {
            'centroids': kmeans.centroids,
            'inertia': kmeans.inertia,
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels)
        }
    
    return results


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(k=4, init_method='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Visualize results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Centroids')
    plt.title('K-Means Clustering Results')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.legend()
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    results = find_optimal_clusters(X_scaled, k_range=range(1, 11))
    
    # Plot results
    plot_elbow_curve(results)
    
    # Print evaluation metrics
    print(f"\nClustering Results:")
    print(f"Inertia: {kmeans.inertia:.2f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, labels):.2f}")
    
    # Compare initialization methods
    print("\nComparing initialization methods:")
    comparison_results = compare_initialization_methods(X_scaled, k=4)
    for method, metrics in comparison_results.items():
        print(f"{method.capitalize()} initialization:")
        print(f"  Inertia: {metrics['inertia']:.2f}")
        print(f"  Silhouette Score: {metrics['silhouette']:.4f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")