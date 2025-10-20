"""
Uniform Manifold Approximation and Projection (UMAP) Implementation
==============================================================

This module provides implementations and comparisons of UMAP for 
dimensionality reduction and manifold learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
    import umap.umap_ as umap_sklearn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")
import warnings
warnings.filterwarnings('ignore')


def visualize_umap_results(X, y, title="UMAP Visualization"):
    """
    Visualize UMAP results with class labels.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="UMAP Visualization"
        Title for the plot.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Using t-SNE as alternative.")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method_name = "t-SNE"
    else:
        reducer = umap_sklearn.UMAP(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method_name = "UMAP"
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(f'{method_name} Dimension 1')
    plt.ylabel(f'{method_name} Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return reducer


def compare_dimensionality_reduction(X, y, title="Dimensionality Reduction Comparison"):
    """
    Compare different dimensionality reduction techniques.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="Dimensionality Reduction Comparison"
        Title for the plot.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply different techniques
    results = {}
    
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    results['PCA'] = X_pca
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    results['t-SNE'] = X_tsne
    
    # UMAP (if available)
    if UMAP_AVAILABLE:
        umap_reducer = umap_sklearn.UMAP(n_components=2, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled)
        results['UMAP'] = X_umap
    
    # Plot results
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for i, (method, X_reduced) in enumerate(results.items()):
        scatter = axes[i].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[i].set_xlabel(f'{method} Dimension 1')
        axes[i].set_ylabel(f'{method} Dimension 2')
        axes[i].set_title(f'{method} Projection')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def plot_parameter_sensitivity(X, y, n_neighbors=[5, 15, 30], min_dist=[0.1, 0.5, 0.9]):
    """
    Show how UMAP parameters affect the embedding.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    y : array-like, shape (n_samples,)
        Class labels.
    n_neighbors : list, default=[5, 15, 30]
        UMAP n_neighbors parameter values.
    min_dist : list, default=[0.1, 0.5, 0.9]
        UMAP min_dist parameter values.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available for parameter sensitivity analysis.")
        return
    
    fig, axes = plt.subplots(len(n_neighbors), len(min_dist), figsize=(15, 15))
    
    for i, n_neighbor in enumerate(n_neighbors):
        for j, min_d in enumerate(min_dist):
            # Apply UMAP
            reducer = umap_sklearn.UMAP(n_components=2, n_neighbors=n_neighbor, 
                                      min_dist=min_d, random_state=42)
            X_umap = reducer.fit_transform(X)
            
            # Plot results
            scatter = axes[i, j].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
            axes[i, j].set_xlabel('UMAP Dimension 1')
            axes[i, j].set_ylabel('UMAP Dimension 2')
            axes[i, j].set_title(f'n_neighbors={n_neighbor}, min_dist={min_d}')
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data - digits dataset for better visualization
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Dimensionality Reduction Comparison:")
    print(f"Original dimensions: {X_scaled.shape}")
    
    # Compare different techniques
    print("\nComparing dimensionality reduction techniques...")
    results = compare_dimensionality_reduction(X_scaled[:500], y[:500], 
                                             "Dimensionality Reduction Comparison")
    
    # Visualize UMAP results (if available)
    print("\nGenerating UMAP visualization...")
    if UMAP_AVAILABLE:
        reducer = visualize_umap_results(X_scaled, y, "Digits Dataset UMAP Projection")
    
    # Parameter sensitivity analysis (if UMAP available)
    print("\nAnalyzing parameter sensitivity...")
    if UMAP_AVAILABLE:
        plot_parameter_sensitivity(X_scaled[:500], y[:500])
    
    # Demonstrate with iris dataset
    print("\nIris Dataset Example:")
    X_iris, y_iris = load_iris(return_X_y=True)
    scaler_iris = StandardScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris)
    
    if UMAP_AVAILABLE:
        visualize_umap_results(X_iris_scaled, y_iris, "Iris Dataset UMAP Projection")
    
    print("\nKey Points about UMAP:")
    print("• UMAP is faster than t-SNE and preserves more global structure")
    print("• Two key parameters: n_neighbors (local vs global structure) and min_dist (clustering)")
    print("• Generally produces better separated clusters than t-SNE")
    print("• More scalable to larger datasets than t-SNE")
    print("• Requires the umap-learn package: pip install umap-learn")