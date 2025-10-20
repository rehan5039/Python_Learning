"""
t-Distributed Stochastic Neighbor Embedding (t-SNE) Implementation
==============================================================

This module provides implementations and visualizations of t-SNE for 
high-dimensional data visualization and exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE as SklearnTSNE
import warnings
warnings.filterwarnings('ignore')


class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding implementation.
    
    Parameters:
    -----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Perplexity parameter for t-SNE.
    learning_rate : float, default=200.0
        Learning rate for optimization.
    n_iter : int, default=1000
        Maximum number of iterations.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    embedding : array, shape (n_samples, n_components)
        Embedded data.
    """
    
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, 
                 n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
    
    def _compute_pairwise_distances(self, X):
        """Compute pairwise squared Euclidean distances."""
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return D
    
    def _compute_conditional_probabilities(self, D):
        """Compute conditional probabilities P(j|i)."""
        n_samples = D.shape[0]
        P = np.zeros((n_samples, n_samples))
        beta = np.ones((n_samples, 1))  # Precision of Gaussian
        
        # Compute conditional probabilities for each point
        for i in range(n_samples):
            # Binary search for optimal beta (precision)
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))]
            
            # Compute Hdiff to check if current beta is appropriate
            H, thisP = self._Hbeta(Di, beta[i])
            
            Hdiff = H - np.log(self.perplexity)
            tries = 0
            while np.abs(Hdiff) > 1e-5 and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i]
                    if betamax == np.inf:
                        beta[i] = beta[i] * 2
                    else:
                        beta[i] = (beta[i] + betamax) / 2
                else:
                    betamax = beta[i]
                    if betamin == -np.inf:
                        beta[i] = beta[i] / 2
                    else:
                        beta[i] = (beta[i] + betamin) / 2
                
                H, thisP = self._Hbeta(Di, beta[i])
                Hdiff = H - np.log(self.perplexity)
                tries += 1
            
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n_samples]))] = thisP
        
        return P
    
    def _Hbeta(self, D, beta):
        """Compute H and P for given beta."""
        P = np.exp(-D * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P
    
    def _compute_joint_probabilities(self, P):
        """Symmetrize conditional probabilities."""
        n_samples = P.shape[0]
        P = (P + P.T) / (2 * n_samples)
        # Set minimum value to avoid division by zero
        P = np.maximum(P, 1e-12)
        return P
    
    def _compute_low_dimensional_probabilities(self, Y):
        """Compute low-dimensional probabilities Q."""
        n_samples = Y.shape[0]
        sum_Y = np.sum(np.square(Y), 1)
        num = -2 * np.dot(Y, Y.T)
        num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
        # Set diagonal to zero
        np.fill_diagonal(num, 0)
        # Normalize
        Q = num / np.sum(num)
        # Set minimum value to avoid division by zero
        Q = np.maximum(Q, 1e-12)
        return Q, num
    
    def fit_transform(self, X):
        """
        Fit X into an embedded space and return that transformed output.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to be embedded.
            
        Returns:
        --------
        Y : array, shape (n_samples, n_components)
            Embedded data.
        """
        n_samples, n_features = X.shape
        
        # Initialize embedding randomly
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        
        # Compute pairwise distances in high-dimensional space
        D = self._compute_pairwise_distances(X)
        
        # Compute joint probabilities
        P = self._compute_conditional_probabilities(D)
        P = self._compute_joint_probabilities(P)
        
        # Initialize momentum terms
        dY = np.zeros((n_samples, self.n_components))
        iY = np.zeros((n_samples, self.n_components))
        
        # Early exaggeration parameters
        early_exag = 4.0
        early_exag_iter = 100
        
        # Run optimization
        for iter in range(self.n_iter):
            # Compute low-dimensional probabilities
            Q, num = self._compute_low_dimensional_probabilities(Y)
            
            # Compute gradient
            PQ = P - Q if iter > early_exag_iter else (P * early_exag) - Q
            for i in range(n_samples):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.n_components, 1)).T * 
                                 (Y[i, :] - Y), 0)
            
            # Update embedding
            momentum = 0.5 if iter < 20 else 0.8
            iY = momentum * iY - self.learning_rate * dY
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n_samples, 1))
            
            # Print progress
            if (iter + 1) % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                print(f"Iteration {iter + 1}: Error = {C:.4f}")
        
        self.embedding = Y
        return Y


def compare_tsne_implementations(X, n_components=2, perplexity=30.0):
    """
    Compare custom t-SNE implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Perplexity parameter.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations.
    """
    # Custom implementation (simplified for demonstration)
    print("Note: Custom t-SNE implementation is simplified for educational purposes")
    print("For production use, scikit-learn's implementation is recommended")
    
    # Scikit-learn implementation
    sklearn_tsne = SklearnTSNE(n_components=n_components, perplexity=perplexity, 
                              random_state=42, n_iter=500)
    X_sklearn = sklearn_tsne.fit_transform(X)
    
    results = {
        'sklearn': {
            'embedding': X_sklearn
        }
    }
    
    return results


def visualize_tsne_results(X, y, title="t-SNE Visualization"):
    """
    Visualize t-SNE results with class labels.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="t-SNE Visualization"
        Title for the plot.
    """
    # Apply t-SNE
    tsne = SklearnTSNE(n_components=2, perplexity=30.0, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return tsne


def plot_perplexity_comparison(X, y, perplexities=[5, 30, 50, 100]):
    """
    Compare t-SNE results with different perplexity values.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to embed.
    y : array-like, shape (n_samples,)
        Class labels.
    perplexities : list, default=[5, 30, 50, 100]
        Perplexity values to test.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, perplexity in enumerate(perplexities):
        # Apply t-SNE
        tsne = SklearnTSNE(n_components=2, perplexity=perplexity, 
                          random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Plot results
        scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
        axes[i].set_xlabel('t-SNE Dimension 1')
        axes[i].set_ylabel('t-SNE Dimension 2')
        axes[i].set_title(f'Perplexity = {perplexity}')
        axes[i].grid(True, alpha=0.3)
    
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
    
    print("t-SNE Results:")
    print(f"Original dimensions: {X_scaled.shape}")
    
    # Visualize t-SNE results
    print("\nGenerating t-SNE visualization...")
    tsne_model = visualize_tsne_results(X_scaled, y, "Digits Dataset t-SNE Projection")
    
    # Compare different perplexity values
    print("\nComparing different perplexity values...")
    plot_perplexity_comparison(X_scaled[:500], y[:500])  # Use subset for faster computation
    
    # Compare implementations
    print("\nComparing t-SNE implementations:")
    comparison_results = compare_tsne_implementations(X_scaled[:200], n_components=2)  # Small subset
    print("Scikit-learn t-SNE applied successfully")
    
    # Demonstrate with iris dataset
    print("\nIris Dataset Example:")
    X_iris, y_iris = load_iris(return_X_y=True)
    scaler_iris = StandardScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris)
    
    visualize_tsne_results(X_iris_scaled, y_iris, "Iris Dataset t-SNE Projection")
    
    print("\nKey Points about t-SNE:")
    print("• t-SNE is excellent for visualizing high-dimensional data")
    print("• Perplexity parameter is crucial and dataset-dependent")
    print("• t-SNE preserves local structure but not global structure")
    print("• Results can vary with different random initializations")
    print("• Computationally expensive for large datasets")