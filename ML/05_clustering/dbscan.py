"""
DBSCAN Clustering Implementation
=============================

This module provides a comprehensive implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm with visualization and parameter optimization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.
    
    Parameters:
    -----------
    eps : float, default=0.5
        Maximum distance between two samples for one to be considered as in the neighborhood of the other
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered as a core point
    metric : str, default='euclidean'
        Distance metric to use
    
    Attributes:
    -----------
    labels : array, shape (n_samples,)
        Cluster labels for each point. Noisy samples are given the label -1
    core_sample_indices : array
        Indices of core samples
    components : array
        Copy of each core sample found by training
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
    
    def _get_neighbors(self, X, point_idx):
        """Get neighbors of a point within eps distance."""
        neighbors = []
        for i, point in enumerate(X):
            if np.linalg.norm(X[point_idx] - point) < self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, X):
        """
        Perform DBSCAN clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training instances to cluster
        """
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)  # Initialize all points as noise
        cluster_id = 0
        
        # For each point, find its neighbors
        neighbors_list = []
        for i in range(n_samples):
            neighbors = self._get_neighbors(X, i)
            neighbors_list.append(neighbors)
        
        # For each point, if not yet processed
        for i in range(n_samples):
            # Skip if already processed
            if self.labels[i] != -1:
                continue
                
            # Find neighbors
            neighbors = neighbors_list[i]
            
            # If not enough neighbors, mark as noise
            if len(neighbors) < self.min_samples:
                continue
                
            # Create new cluster
            self.labels[i] = cluster_id
            
            # Process neighbors
            seeds = set(neighbors)
            seeds.remove(i)
            
            while seeds:
                current_point = seeds.pop()
                
                # If point was noise, change to border point
                if self.labels[current_point] == -1:
                    self.labels[current_point] = cluster_id
                    
                # If point already processed, skip
                if self.labels[current_point] != -1:
                    continue
                    
                # Mark point as part of cluster
                self.labels[current_point] = cluster_id
                
                # Find neighbors of current point
                current_neighbors = neighbors_list[current_point]
                
                # If current point is core point, add its neighbors to seeds
                if len(current_neighbors) >= self.min_samples:
                    seeds.update(current_neighbors)
            
            cluster_id += 1
        
        # Store core sample indices
        self.core_sample_indices = np.where(np.bincount(self.labels[self.labels != -1]) >= self.min_samples)[0]
        
        return self
    
    def fit_predict(self, X):
        """
        Perform DBSCAN clustering and return cluster labels.
        
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


def plot_k_distance_graph(X, k=4, metric='euclidean'):
    """
    Plot k-distance graph to help determine optimal eps parameter.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to analyze
    k : int, default=4
        Number of neighbors to consider
    metric : str, default='euclidean'
        Distance metric to use
        
    Returns:
    --------
    distances : array
        Sorted k-distances
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # Sort distances
    k_distances = np.sort(distances[:, -1])[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-distance')
    plt.title('K-distance Graph for DBSCAN Parameter Selection')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return k_distances


def optimize_dbscan_parameters(X, eps_range=None, min_samples_range=None):
    """
    Optimize DBSCAN parameters using grid search and silhouette score.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    eps_range : array-like, optional
        Range of eps values to test
    min_samples_range : array-like, optional
        Range of min_samples values to test
        
    Returns:
    --------
    best_params : dict
        Best parameters found
    best_score : float
        Best silhouette score
    """
    if eps_range is None:
        eps_range = np.arange(0.1, 2.0, 0.1)
    
    if min_samples_range is None:
        min_samples_range = range(2, 11)
    
    best_score = -1
    best_params = {'eps': 0.5, 'min_samples': 5}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Only evaluate if we have more than one cluster and not all noise
                if len(set(labels)) > 1 and -1 not in labels:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                elif len(set(labels)) > 1:  # Has clusters but also noise
                    # Calculate silhouette ignoring noise points
                    mask = labels != -1
                    if np.sum(mask) > 1:  # Need at least 2 non-noise points
                        score = silhouette_score(X[mask], labels[mask])
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
            except:
                continue
    
    return best_params, best_score


def compare_dbscan_implementations(X, eps=0.5, min_samples=5):
    """
    Compare custom DBSCAN implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    eps : float, default=0.5
        Epsilon parameter
    min_samples : int, default=5
        Minimum samples parameter
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations
    """
    # Custom implementation
    custom_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    custom_labels = custom_dbscan.fit_predict(X)
    
    # Scikit-learn implementation
    sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    
    # Calculate metrics if possible
    custom_clusters = len(set(custom_labels)) - (1 if -1 in custom_labels else 0)
    sklearn_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    
    results = {
        'custom': {
            'labels': custom_labels,
            'n_clusters': custom_clusters,
            'n_noise': list(custom_labels).count(-1)
        },
        'sklearn': {
            'labels': sklearn_labels,
            'n_clusters': sklearn_clusters,
            'n_noise': list(sklearn_labels).count(-1)
        }
    }
    
    # Calculate agreement if both found clusters
    if custom_clusters > 0 and sklearn_clusters > 0:
        try:
            ari = adjusted_rand_score(custom_labels, sklearn_labels)
            results['adjusted_rand_score'] = ari
        except:
            results['adjusted_rand_score'] = None
    
    return results


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data with noise
    X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Add some noise points
    noise = np.random.uniform(low=-10, high=10, size=(20, 2))
    X_with_noise = np.vstack([X_blobs, noise])
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_noise)
    
    # Plot k-distance graph to help choose eps
    print("Plotting k-distance graph to help determine eps parameter...")
    k_distances = plot_k_distance_graph(X_scaled, k=4)
    
    # Apply DBSCAN clustering
    dbscan = SklearnDBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Original data with noise
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='gray', alpha=0.6)
    plt.title('Original Data with Noise')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.grid(True, alpha=0.3)
    
    # DBSCAN results
    plt.subplot(1, 2, 2)
    # Plot noise points in black
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
                   c='black', marker='x', label='Noise', s=50)
    
    # Plot clustered points
    clustered_mask = labels != -1
    if np.any(clustered_mask):
        scatter = plt.scatter(X_scaled[clustered_mask, 0], X_scaled[clustered_mask, 1], 
                             c=labels[clustered_mask], cmap='viridis')
        plt.colorbar(scatter)
    
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print clustering statistics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\nDBSCAN Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Calculate silhouette score if we have clusters
    if n_clusters > 1 and n_noise < len(labels):
        # Calculate silhouette ignoring noise points
        mask = labels != -1
        if np.sum(mask) > 1:
            silhouette_avg = silhouette_score(X_scaled[mask], labels[mask])
            print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
    
    # Optimize parameters
    print("\nOptimizing DBSCAN parameters...")
    best_params, best_score = optimize_dbscan_parameters(X_scaled)
    print(f"Best parameters: eps={best_params['eps']:.2f}, min_samples={best_params['min_samples']}")
    print(f"Best silhouette score: {best_score:.4f}")
    
    # Compare implementations
    print("\nComparing implementations...")
    comparison_results = compare_dbscan_implementations(X_scaled, eps=0.3, min_samples=5)
    print(f"Custom implementation: {comparison_results['custom']['n_clusters']} clusters, "
          f"{comparison_results['custom']['n_noise']} noise points")
    print(f"Scikit-learn implementation: {comparison_results['sklearn']['n_clusters']} clusters, "
          f"{comparison_results['sklearn']['n_noise']} noise points")
    
    if 'adjusted_rand_score' in comparison_results:
        print(f"Agreement between implementations: {comparison_results['adjusted_rand_score']:.4f}")