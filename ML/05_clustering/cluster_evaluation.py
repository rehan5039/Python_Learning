"""
Cluster Evaluation Metrics and Techniques
=====================================

This module provides comprehensive implementations of cluster evaluation metrics
and techniques for assessing clustering quality and determining optimal parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (silhouette_score, silhouette_samples, 
                           adjusted_rand_score, normalized_mutual_info_score,
                           calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


def plot_silhouette_analysis(X, labels, metric='euclidean'):
    """
    Plot silhouette analysis for clustering results.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
    metric : str, default='euclidean'
        Distance metric to use
    """
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)
    
    # Plot silhouette analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    y_lower = 10
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax.set_xlabel('Silhouette Coefficient Values')
    ax.set_ylabel('Cluster Label')
    
    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Average: {silhouette_avg:.3f}')
    
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    
    plt.title(f'Silhouette Analysis for {n_clusters} Clusters')
    plt.tight_layout()
    plt.show()
    
    return silhouette_avg


def calculate_internal_metrics(X, labels):
    """
    Calculate various internal clustering metrics.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    labels : array-like, shape (n_samples,)
        Cluster labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    # Remove noise points for metric calculation if present
    mask = labels != -1
    if np.any(mask) and np.sum(mask) > 1:
        X_filtered = X[mask]
        labels_filtered = labels[mask]
    else:
        X_filtered = X
        labels_filtered = labels
    
    metrics = {}
    
    # Only calculate metrics if we have more than one cluster
    unique_labels = set(labels_filtered)
    if len(unique_labels) > 1:
        # Silhouette Score
        try:
            metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
        except:
            metrics['silhouette_score'] = None
        
        # Calinski-Harabasz Index
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_filtered, labels_filtered)
        except:
            metrics['calinski_harabasz_score'] = None
        
        # Davies-Bouldin Index
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
        except:
            metrics['davies_bouldin_score'] = None
    
    # Count clusters and noise points
    metrics['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
    metrics['n_noise_points'] = list(labels).count(-1) if -1 in labels else 0
    
    return metrics


def calculate_external_metrics(true_labels, predicted_labels):
    """
    Calculate external clustering metrics when true labels are available.
    
    Parameters:
    -----------
    true_labels : array-like, shape (n_samples,)
        True cluster labels
    predicted_labels : array-like, shape (n_samples,)
        Predicted cluster labels
        
    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated external metrics
    """
    metrics = {}
    
    # Adjusted Rand Index
    try:
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
    except:
        metrics['adjusted_rand_score'] = None
    
    # Normalized Mutual Information
    try:
        metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, predicted_labels)
    except:
        metrics['normalized_mutual_info_score'] = None
    
    return metrics


def gap_statistic(X, clustering_func, k_range=range(1, 11), n_refs=10, random_state=42):
    """
    Calculate gap statistic to determine optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    clustering_func : callable
        Function that performs clustering and returns labels
    k_range : range, default=range(1, 11)
        Range of k values to test
    n_refs : int, default=10
        Number of reference datasets to generate
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing gap statistics for each k
    """
    np.random.seed(random_state)
    
    gaps = []
    results = []
    
    for k in k_range:
        # Perform clustering on actual data
        if k == 1:
            labels = np.zeros(X.shape[0])
        else:
            labels = clustering_func(X, k)
        
        # Calculate within-cluster dispersion
        wk = _calculate_wk(X, labels)
        
        # Generate reference datasets and calculate their dispersions
        wk_refs = []
        for _ in range(n_refs):
            # Generate random reference data
            X_ref = np.random.uniform(X.min(), X.max(), X.shape)
            if k == 1:
                ref_labels = np.zeros(X_ref.shape[0])
            else:
                ref_labels = clustering_func(X_ref, k)
            wk_refs.append(_calculate_wk(X_ref, ref_labels))
        
        # Calculate gap statistic
        gap = np.log(np.mean(wk_refs)) - np.log(wk)
        gaps.append(gap)
        
        # Calculate standard deviation
        sdk = np.sqrt(1 + 1.0/n_refs) * np.std(np.log(wk_refs))
        results.append({
            'k': k,
            'wk': wk,
            'gap': gap,
            'sdk': sdk
        })
    
    return results


def _calculate_wk(X, labels):
    """Calculate within-cluster dispersion."""
    unique_labels = np.unique(labels)
    wk = 0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            # Calculate cluster centroid
            centroid = np.mean(cluster_points, axis=0)
            # Calculate sum of squared distances to centroid
            distances = np.sum((cluster_points - centroid) ** 2)
            wk += distances
    
    return wk


def elbow_method(X, k_range=range(1, 11), random_state=42):
    """
    Apply elbow method to find optimal number of clusters.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    k_range : range, default=range(1, 11)
        Range of k values to test
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    inertias : list
        Inertia values for each k
    """
    inertias = []
    
    for k in k_range:
        if k == 1:
            # For k=1, inertia is total variance
            inertia = np.sum((X - np.mean(X, axis=0)) ** 2)
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(X)
            inertia = kmeans.inertia_
        inertias.append(inertia)
    
    return list(k_range), inertias


def plot_validation_curves(k_range, inertias=None, gaps=None, silhouette_scores=None):
    """
    Plot validation curves for cluster validation.
    
    Parameters:
    -----------
    k_range : array-like
        Range of k values tested
    inertias : list, optional
        Inertia values for elbow method
    gaps : list, optional
        Gap statistics
    silhouette_scores : list, optional
        Silhouette scores
    """
    n_plots = sum([inertias is not None, gaps is not None, silhouette_scores is not None])
    
    if n_plots == 0:
        print("No data provided for plotting")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Elbow method plot
    if inertias is not None:
        axes[plot_idx].plot(k_range, inertias, 'bo-')
        axes[plot_idx].set_xlabel('Number of Clusters (k)')
        axes[plot_idx].set_ylabel('Inertia')
        axes[plot_idx].set_title('Elbow Method')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Gap statistic plot
    if gaps is not None:
        axes[plot_idx].plot(k_range, gaps, 'ro-')
        axes[plot_idx].set_xlabel('Number of Clusters (k)')
        axes[plot_idx].set_ylabel('Gap Statistic')
        axes[plot_idx].set_title('Gap Statistic Method')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Silhouette score plot
    if silhouette_scores is not None:
        axes[plot_idx].plot(k_range[1:], silhouette_scores[1:], 'go-')  # Skip k=1
        axes[plot_idx].set_xlabel('Number of Clusters (k)')
        axes[plot_idx].set_ylabel('Silhouette Score')
        axes[plot_idx].set_title('Silhouette Analysis')
        axes[plot_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def cluster_stability_analysis(X, clustering_func, k, n_bootstrap=10):
    """
    Assess cluster stability through bootstrap sampling.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    clustering_func : callable
        Function that performs clustering and returns labels
    k : int
        Number of clusters
    n_bootstrap : int, default=10
        Number of bootstrap samples
        
    Returns:
    --------
    stability_score : float
        Average stability score across bootstrap samples
    """
    n_samples = X.shape[0]
    ari_scores = []
    
    # Get original clustering
    original_labels = clustering_func(X, k)
    
    # Bootstrap sampling
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X[indices]
        
        # Cluster bootstrap sample
        bootstrap_labels = clustering_func(X_bootstrap, k)
        
        # Map bootstrap labels back to original indices
        # This is a simplified approach - in practice, you might need more sophisticated matching
        if len(bootstrap_labels) == len(original_labels):
            ari = adjusted_rand_score(original_labels, bootstrap_labels)
            ari_scores.append(ari)
    
    return np.mean(ari_scores) if ari_scores else 0.0


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate internal metrics
    print("Internal Clustering Metrics:")
    internal_metrics = calculate_internal_metrics(X_scaled, labels)
    for metric, value in internal_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: Not available")
    
    # Calculate external metrics (since we have true labels)
    print("\nExternal Clustering Metrics:")
    external_metrics = calculate_external_metrics(y_true, labels)
    for metric, value in external_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: Not available")
    
    # Silhouette analysis
    print("\nGenerating silhouette analysis plot...")
    silhouette_avg = plot_silhouette_analysis(X_scaled, labels)
    print(f"Average silhouette score: {silhouette_avg:.4f}")
    
    # Elbow method
    print("\nApplying elbow method...")
    k_range, inertias = elbow_method(X_scaled, k_range=range(1, 11))
    
    # Silhouette analysis for different k values
    print("Calculating silhouette scores for different k values...")
    silhouette_scores = []
    for k in k_range:
        if k == 1:
            silhouette_scores.append(0)  # Silhouette not defined for k=1
        else:
            kmeans_temp = KMeans(n_clusters=k, random_state=42)
            labels_temp = kmeans_temp.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels_temp)
            silhouette_scores.append(score)
    
    # Plot validation curves
    print("Plotting validation curves...")
    plot_validation_curves(k_range, inertias=inertias, silhouette_scores=silhouette_scores)
    
    # Stability analysis
    print("\nPerforming cluster stability analysis...")
    def kmeans_clustering(X, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        return kmeans.fit_predict(X)
    
    stability_score = cluster_stability_analysis(X_scaled, kmeans_clustering, k=4, n_bootstrap=5)
    print(f"Cluster stability score: {stability_score:.4f}")
    
    # Demonstrate with DBSCAN (different number of clusters each time)
    print("\nDemonstrating with DBSCAN...")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    dbscan_metrics = calculate_internal_metrics(X_scaled, dbscan_labels)
    print("DBSCAN Metrics:")
    for metric, value in dbscan_metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: Not available")