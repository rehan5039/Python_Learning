"""
Principal Component Analysis (PCA) Implementation
============================================

This module provides a comprehensive implementation of Principal Component Analysis
with mathematical foundations, visualization capabilities, and practical applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class PCA:
    """
    Principal Component Analysis implementation.
    
    Parameters:
    -----------
    n_components : int, default=None
        Number of components to keep. If None, keep all components.
    svd_solver : str, default='auto'
        Solver to use for computation ('auto', 'full', 'arpack', 'randomized').
    
    Attributes:
    -----------
    components : array, shape (n_components, n_features)
        Principal components (principal axes in feature space).
    explained_variance : array, shape (n_components,)
        Amount of variance explained by each of the selected components.
    explained_variance_ratio : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    mean : array, shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    """
    
    def __init__(self, n_components=None, svd_solver='auto'):
        self.n_components = n_components
        self.svd_solver = svd_solver
    
    def fit(self, X):
        """
        Fit the PCA model with X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        """
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
        
        self.components = eigenvectors.T
        self.explained_variance = eigenvalues
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction to X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data.
            
        Returns:
        --------
        X_new : array, shape (n_samples, n_components)
            Transformed values.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)
    
    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        X_new : array, shape (n_samples, n_components)
            Transformed values.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_components)
            Data in transformed space.
            
        Returns:
        --------
        X_original : array, shape (n_samples, n_features)
            Data in original space.
        """
        return np.dot(X, self.components) + self.mean


def plot_pca_explained_variance(pca, title="PCA Explained Variance"):
    """
    Plot explained variance and cumulative explained variance.
    
    Parameters:
    -----------
    pca : PCA object
        Fitted PCA model.
    title : str, default="PCA Explained Variance"
        Title for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Explained variance
    ax1.bar(range(1, len(pca.explained_variance_ratio) + 1), pca.explained_variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'{title} - Individual')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    cumsum_variance = np.cumsum(pca.explained_variance_ratio)
    ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title(f'{title} - Cumulative')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def find_optimal_components(X, variance_threshold=0.95):
    """
    Find optimal number of components to explain desired variance.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to analyze.
    variance_threshold : float, default=0.95
        Desired variance threshold.
        
    Returns:
    --------
    n_components : int
        Optimal number of components.
    """
    pca = SklearnPCA()
    pca.fit(X)
    
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
    
    return n_components


def compare_pca_implementations(X, n_components=2):
    """
    Compare custom PCA implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to transform.
    n_components : int, default=2
        Number of components.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Custom implementation
    custom_pca = PCA(n_components=n_components)
    X_custom = custom_pca.fit_transform(X)
    
    # Scikit-learn implementation
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X)
    
    # Calculate reconstruction error
    X_reconstructed_custom = custom_pca.inverse_transform(X_custom)
    X_reconstructed_sklearn = sklearn_pca.inverse_transform(X_sklearn)
    
    mse_custom = mean_squared_error(X, X_reconstructed_custom)
    mse_sklearn = mean_squared_error(X, X_reconstructed_sklearn)
    
    results = {
        'custom': {
            'transformed_data': X_custom,
            'explained_variance_ratio': custom_pca.explained_variance_ratio,
            'mse': mse_custom
        },
        'sklearn': {
            'transformed_data': X_sklearn,
            'explained_variance_ratio': sklearn_pca.explained_variance_ratio_,
            'mse': mse_sklearn
        }
    }
    
    return results


def visualize_pca_2d(X, y, title="PCA Visualization"):
    """
    Visualize 2D PCA results with class labels.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to transform.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="PCA Visualization"
        Title for the plot.
    """
    # Apply PCA
    pca = SklearnPCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return pca


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data
    X, y = load_iris(return_X_y=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Print results
    print("PCA Results:")
    print(f"Original dimensions: {X_scaled.shape}")
    print(f"Reduced dimensions: {X_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio):.1%}")
    
    # Visualize explained variance
    plot_pca_explained_variance(pca, "Iris Dataset PCA")
    
    # Visualize 2D projection
    visualize_pca_2d(X_scaled, y, "Iris Dataset PCA Projection")
    
    # Find optimal number of components
    n_components = find_optimal_components(X_scaled, variance_threshold=0.95)
    print(f"\nNumber of components to explain 95% variance: {n_components}")
    
    # Compare implementations
    print("\nComparing PCA implementations:")
    comparison_results = compare_pca_implementations(X_scaled, n_components=2)
    print(f"Custom PCA MSE: {comparison_results['custom']['mse']:.6f}")
    print(f"Scikit-learn PCA MSE: {comparison_results['sklearn']['mse']:.6f}")
    print(f"Custom PCA explained variance: {comparison_results['custom']['explained_variance_ratio']}")
    print(f"Scikit-learn PCA explained variance: {comparison_results['sklearn']['explained_variance_ratio']}")
    
    # Demonstrate reconstruction
    pca_full = PCA()
    pca_full.fit(X_scaled)
    X_reduced = pca_full.transform(X_scaled)
    X_reconstructed = pca_full.inverse_transform(X_reduced)
    
    reconstruction_error = mean_squared_error(X_scaled, X_reconstructed)
    print(f"\nFull PCA reconstruction error: {reconstruction_error:.6f}")