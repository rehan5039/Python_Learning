"""
Chapter 5: Clustering - Practice Problems
=====================================

This file contains practice problems for clustering algorithms with solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


# Problem 1: K-Means Implementation from Scratch
def problem_1_kmeans_implementation():
    """
    Problem: Implement K-Means clustering from scratch and compare with scikit-learn.
    
    Requirements:
    - Implement K-Means without using scikit-learn's KMeans
    - Compare results with scikit-learn implementation
    - Visualize the clustering results
    """
    print("Problem 1: K-Means Implementation from Scratch")
    print("=" * 50)
    
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Custom K-Means implementation
    class CustomKMeans:
        def __init__(self, k=3, max_iters=100, tol=1e-4):
            self.k = k
            self.max_iters = max_iters
            self.tol = tol
        
        def fit(self, X):
            # Initialize centroids randomly
            n_samples, n_features = X.shape
            self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
            
            # Iteratively update centroids
            for i in range(self.max_iters):
                # Assign points to closest centroids
                distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                self.labels = np.argmin(distances, axis=0)
                
                # Update centroids
                new_centroids = np.array([X[self.labels == j].mean(axis=0) for j in range(self.k)])
                
                # Check for convergence
                if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                    break
                    
                self.centroids = new_centroids
            
            return self
        
        def predict(self, X):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            return np.argmin(distances, axis=0)
    
    # Apply custom implementation
    custom_kmeans = CustomKMeans(k=4)
    custom_kmeans.fit(X_scaled)
    custom_labels = custom_kmeans.labels
    
    # Apply scikit-learn implementation
    sklearn_kmeans = KMeans(n_clusters=4, random_state=42)
    sklearn_labels = sklearn_kmeans.fit_predict(X_scaled)
    
    # Compare results
    custom_silhouette = silhouette_score(X_scaled, custom_labels)
    sklearn_silhouette = silhouette_score(X_scaled, sklearn_labels)
    
    print(f"Custom K-Means Silhouette Score: {custom_silhouette:.4f}")
    print(f"Scikit-learn K-Means Silhouette Score: {sklearn_silhouette:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original data
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c='gray', alpha=0.7)
    axes[0].set_title('Original Data')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # Custom K-Means
    axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=custom_labels, cmap='viridis')
    axes[1].scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
    axes[1].set_title(f'Custom K-Means (Score: {custom_silhouette:.3f})')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].grid(True, alpha=0.3)
    
    # Scikit-learn K-Means
    axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=sklearn_labels, cmap='viridis')
    axes[2].scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
    axes[2].set_title(f'Scikit-learn K-Means (Score: {sklearn_silhouette:.3f})')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return custom_kmeans, sklearn_kmeans


# Problem 2: Customer Segmentation
def problem_2_customer_segmentation():
    """
    Problem: Segment customers based on purchasing behavior using clustering.
    
    Requirements:
    - Generate synthetic customer data
    - Apply multiple clustering algorithms
    - Interpret the segments
    - Visualize the results
    """
    print("\nProblem 2: Customer Segmentation")
    print("=" * 40)
    
    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 500
    
    # Features: Annual Spending, Frequency of Purchases, Average Order Value, Years as Customer
    annual_spending = np.random.gamma(2, 5000, n_customers)
    purchase_frequency = np.random.poisson(10, n_customers)
    avg_order_value = np.random.normal(100, 30, n_customers)
    years_customer = np.random.exponential(3, n_customers)
    
    # Create feature matrix
    X = np.column_stack([annual_spending, purchase_frequency, avg_order_value, years_customer])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Customer Segmentation Silhouette Score: {silhouette_avg:.4f}")
    
    # Interpret segments
    segment_means = []
    for i in range(5):
        segment_data = X[labels == i]
        segment_means.append({
            'segment': i,
            'size': len(segment_data),
            'annual_spending': np.mean(segment_data[:, 0]),
            'purchase_frequency': np.mean(segment_data[:, 1]),
            'avg_order_value': np.mean(segment_data[:, 2]),
            'years_customer': np.mean(segment_data[:, 3])
        })
    
    print("\nCustomer Segments:")
    for segment in segment_means:
        print(f"Segment {segment['segment']}: {segment['size']} customers")
        print(f"  Avg Annual Spending: ${segment['annual_spending']:.0f}")
        print(f"  Avg Purchase Frequency: {segment['purchase_frequency']:.1f} purchases/year")
        print(f"  Avg Order Value: ${segment['avg_order_value']:.0f}")
        print(f"  Avg Years as Customer: {segment['years_customer']:.1f} years")
        print()
    
    # Visualize using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', alpha=0.6)
    plt.title('Customer Data (PCA Projection)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.grid(True, alpha=0.3)
    
    # Clustered data
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f'Customer Segments (Score: {silhouette_avg:.3f})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return kmeans, labels, segment_means


# Problem 3: Image Compression using K-Means
def problem_3_image_compression():
    """
    Problem: Use K-Means for color quantization to compress images.
    
    Requirements:
    - Load an image
    - Apply K-Means to reduce color palette
    - Compare original and compressed images
    - Measure compression ratio
    """
    print("\nProblem 3: Image Compression using K-Means")
    print("=" * 45)
    
    # Create a sample image (in practice, you would load a real image)
    # For demonstration, we'll create a synthetic image
    np.random.seed(42)
    height, width = 100, 100
    # Create an image with random colors
    original_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Reshape image to be a list of pixels
    pixels = original_image.reshape(-1, 3)
    
    # Apply K-Means to find color palette
    n_colors = 16  # Reduce to 16 colors
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Get the colors (cluster centers)
    palette = kmeans.cluster_centers_.astype(int)
    
    # Map each pixel to its nearest color in the palette
    labels = kmeans.predict(pixels)
    compressed_pixels = palette[labels]
    compressed_image = compressed_pixels.reshape(original_image.shape)
    
    # Calculate compression ratio
    original_size = original_image.size
    # In compressed image, we store palette + indices
    compressed_size = (n_colors * 3) + len(labels)  # Palette + indices
    compression_ratio = original_size / compressed_size
    
    print(f"Original image size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Compressed image
    axes[1].imshow(compressed_image)
    axes[1].set_title(f'Compressed Image ({n_colors} colors)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show color palette
    plt.figure(figsize=(12, 2))
    for i, color in enumerate(palette):
        plt.bar(i, 1, color=color/255.0, edgecolor='black')
    plt.title('Color Palette')
    plt.xlabel('Color Index')
    plt.ylabel('Color')
    plt.xticks(range(n_colors))
    plt.yticks([])
    plt.show()
    
    return kmeans, compressed_image, compression_ratio


# Problem 4: Hierarchical Clustering Analysis
def problem_4_hierarchical_analysis():
    """
    Problem: Compare different linkage methods in hierarchical clustering.
    
    Requirements:
    - Generate data with different cluster shapes
    - Apply hierarchical clustering with different linkage methods
    - Compare results visually and quantitatively
    """
    print("\nProblem 4: Hierarchical Clustering Analysis")
    print("=" * 45)
    
    # Generate data with non-spherical clusters (moons dataset)
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply different linkage methods
    linkage_methods = ['ward', 'complete', 'average', 'single']
    results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(linkage_methods):
        # Apply hierarchical clustering
        if method == 'ward':
            clustering = AgglomerativeClustering(n_clusters=2, linkage=method)
        else:
            clustering = AgglomerativeClustering(n_clusters=2, linkage=method, metric='euclidean')
        
        labels = clustering.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        
        results[method] = {
            'labels': labels,
            'silhouette_score': silhouette
        }
        
        # Visualize results
        plt.subplot(2, 2, i+1)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
        plt.title(f'{method.capitalize()} Linkage (Score: {silhouette:.3f})')
        plt.xlabel('Feature 1 (Standardized)')
        plt.ylabel('Feature 2 (Standardized)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison
    print("Linkage Method Comparison:")
    for method, result in results.items():
        print(f"  {method.capitalize()} linkage: {result['silhouette_score']:.4f}")
    
    return results


# Problem 5: DBSCAN Parameter Tuning
def problem_5_dbscan_tuning():
    """
    Problem: Find optimal parameters for DBSCAN clustering.
    
    Requirements:
    - Generate data with noise points
    - Use k-distance graph to find optimal eps
    - Systematically tune parameters
    - Evaluate results
    """
    print("\nProblem 5: DBSCAN Parameter Tuning")
    print("=" * 35)
    
    # Generate data with noise
    X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    noise = np.random.uniform(low=-10, high=10, size=(30, 2))
    X_with_noise = np.vstack([X_blobs, noise])
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_noise)
    
    # Plot k-distance graph to help choose eps
    from sklearn.neighbors import NearestNeighbors
    k = 4
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])[::-1]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(k_distances)), k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-distance')
    plt.title('K-distance Graph for DBSCAN Parameter Selection')
    plt.grid(True, alpha=0.3)
    
    # Test different parameter combinations
    eps_values = [0.2, 0.3, 0.4, 0.5]
    min_samples_values = [3, 5, 7]
    
    best_score = -1
    best_params = None
    results = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculate silhouette score (excluding noise points)
            if n_clusters > 1 and n_noise < len(labels):
                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette = silhouette_score(X_scaled[mask], labels[mask])
                else:
                    silhouette = 0
            else:
                silhouette = 0
            
            results[(eps, min_samples)] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette
            }
            
            if silhouette > best_score:
                best_score = silhouette
                best_params = (eps, min_samples)
    
    # Apply DBSCAN with best parameters
    best_dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1])
    best_labels = best_dbscan.fit_predict(X_scaled)
    
    # Visualize best result
    plt.subplot(1, 2, 2)
    noise_mask = best_labels == -1
    if np.any(noise_mask):
        plt.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
                   c='black', marker='x', label='Noise', s=50)
    
    clustered_mask = best_labels != -1
    if np.any(clustered_mask):
        scatter = plt.scatter(X_scaled[clustered_mask, 0], X_scaled[clustered_mask, 1], 
                             c=best_labels[clustered_mask], cmap='viridis')
        plt.colorbar(scatter)
    
    plt.title(f'Best DBSCAN Result (eps={best_params[0]}, min_samples={best_params[1]})')
    plt.xlabel('Feature 1 (Standardized)')
    plt.ylabel('Feature 2 (Standardized)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Best parameters: eps={best_params[0]}, min_samples={best_params[1]}")
    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Number of clusters: {results[best_params]['n_clusters']}")
    print(f"Number of noise points: {results[best_params]['n_noise']}")
    
    return best_dbscan, best_params, results


# Main execution
if __name__ == "__main__":
    print("Chapter 5: Clustering - Practice Problems")
    print("========================================")
    
    # Run all problems
    try:
        # Problem 1: K-Means Implementation
        custom_kmeans, sklearn_kmeans = problem_1_kmeans_implementation()
        
        # Problem 2: Customer Segmentation
        kmeans_seg, labels_seg, segments = problem_2_customer_segmentation()
        
        # Problem 3: Image Compression
        kmeans_img, compressed_img, compression_ratio = problem_3_image_compression()
        
        # Problem 4: Hierarchical Clustering Analysis
        hierarchical_results = problem_4_hierarchical_analysis()
        
        # Problem 5: DBSCAN Parameter Tuning
        dbscan_best, best_params, tuning_results = problem_5_dbscan_tuning()
        
        print("\n🎉 All practice problems completed successfully!")
        print("\n📋 Summary of what you've learned:")
        print("  • Implemented K-Means clustering from scratch")
        print("  • Applied clustering to customer segmentation")
        print("  • Used K-Means for image compression")
        print("  • Compared different hierarchical clustering methods")
        print("  • Tuned DBSCAN parameters for optimal results")
        
    except Exception as e:
        print(f"Error running practice problems: {e}")
        print("Please check the implementations and try again.")