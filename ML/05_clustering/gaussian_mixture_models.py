"""
Gaussian Mixture Models Implementation
===================================

This module provides a comprehensive implementation of Gaussian Mixture Models (GMM)
with Expectation-Maximization algorithm, model selection, and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture as SklearnGMM
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


class GaussianMixture:
    """
    Gaussian Mixture Model implementation using Expectation-Maximization algorithm.
    
    Parameters:
    -----------
    n_components : int, default=1
        Number of mixture components
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Tolerance for convergence
    covariance_type : str, default='full'
        Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes:
    -----------
    weights : array, shape (n_components,)
        Weights of each mixture component
    means : array, shape (n_components, n_features)
        Mean parameters for each mixture component
    covariances : array
        Covariance parameters for each mixture component
    converged : bool
        Whether the algorithm converged
    n_iter : int
        Number of iterations run
    """
    
    def __init__(self, n_components=1, max_iter=100, tol=1e-6, 
                 covariance_type='full', random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)
    
    def _initialize_parameters(self, X):
        """Initialize model parameters."""
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
        # Initialize means by randomly selecting data points
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices].copy()
        
        # Initialize covariances
        if self.covariance_type == 'full':
            # Full covariance matrices
            cov_init = np.cov(X.T) + 1e-6 * np.eye(n_features)
            self.covariances = np.array([cov_init for _ in range(self.n_components)])
        elif self.covariance_type == 'diag':
            # Diagonal covariance matrices
            var_init = np.var(X, axis=0) + 1e-6
            self.covariances = np.array([np.diag(var_init) for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            # Spherical covariance (scalar times identity)
            var_init = np.mean(np.var(X, axis=0)) + 1e-6
            self.covariances = np.array([var_init * np.eye(n_features) 
                                       for _ in range(self.n_components)])
        else:  # tied
            # Shared covariance matrix
            cov_init = np.cov(X.T) + 1e-6 * np.eye(n_features)
            self.covariances = np.array([cov_init for _ in range(self.n_components)])
    
    def _e_step(self, X):
        """Expectation step: compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute responsibilities for each component
        for k in range(self.n_components):
            # For numerical stability, we work in log space
            log_prob = multivariate_normal.logpdf(X, self.means[k], self.covariances[k])
            responsibilities[:, k] = np.log(self.weights[k]) + log_prob
        
        # Normalize responsibilities using logsumexp for numerical stability
        max_log_prob = np.max(responsibilities, axis=1, keepdims=True)
        responsibilities = responsibilities - max_log_prob
        responsibilities = np.exp(responsibilities)
        responsibilities = responsibilities / np.sum(responsibilities, axis=1, keepdims=True)
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """Maximization step: update parameters."""
        n_samples, n_features = X.shape
        Nk = np.sum(responsibilities, axis=0) + 1e-10  # Add small value to avoid division by zero
        
        # Update weights
        self.weights = Nk / n_samples
        
        # Update means
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        
        # Update covariances
        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]
                # Add regularization to prevent singular matrices
                self.covariances[k] += 1e-6 * np.eye(n_features)
        elif self.covariance_type == 'diag':
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.diag(np.sum(responsibilities[:, k][:, np.newaxis] * 
                                                   diff**2, axis=0) / Nk[k])
                self.covariances[k] += 1e-6
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.eye(n_features) * np.sum(responsibilities[:, k] * 
                                                                np.sum(diff**2, axis=1)) / \
                                    (n_features * Nk[k])
                self.covariances[k] += 1e-6 * np.eye(n_features)
        else:  # tied
            # Shared covariance matrix
            shared_cov = np.zeros_like(self.covariances[0])
            for k in range(self.n_components):
                diff = X - self.means[k]
                shared_cov += np.dot(responsibilities[:, k] * diff.T, diff)
            shared_cov = shared_cov / n_samples
            shared_cov += 1e-6 * np.eye(n_features)
            self.covariances = np.array([shared_cov for _ in range(self.n_components)])
    
    def _compute_log_likelihood(self, X):
        """Compute log likelihood of the data."""
        log_likelihood = 0
        for i in range(X.shape[0]):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += self.weights[k] * multivariate_normal.pdf(
                    X[i], self.means[k], self.covariances[k])
            log_likelihood += np.log(sample_likelihood + 1e-10)
        return log_likelihood
    
    def fit(self, X):
        """
        Estimate model parameters with the EM algorithm.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        # Initialize parameters
        self._initialize_parameters(X)
        
        prev_log_likelihood = -np.inf
        
        # EM algorithm
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check for convergence
            log_likelihood = self._compute_log_likelihood(X)
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                self.n_iter = iteration + 1
                break
            
            prev_log_likelihood = log_likelihood
        
        if not hasattr(self, 'converged'):
            self.converged = False
            self.n_iter = self.max_iter
        
        return self
    
    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities of each component
        """
        return self._e_step(X)
    
    def predict(self, X):
        """
        Predict the labels for the data samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Predicted labels
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)
    
    def score_samples(self, X):
        """
        Compute the log likelihood of each sample.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        log_likelihood : array, shape (n_samples,)
            Log likelihood of each sample
        """
        log_likelihood = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += self.weights[k] * multivariate_normal.pdf(
                    X[i], self.means[k], self.covariances[k])
            log_likelihood[i] = np.log(sample_likelihood + 1e-10)
        return log_likelihood
    
    def bic(self, X):
        """
        Bayesian Information Criterion for the model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to evaluate
            
        Returns:
        --------
        bic : float
            BIC score
        """
        log_likelihood = np.sum(self.score_samples(X))
        n_params = self._n_parameters()
        n_samples = X.shape[0]
        return -2 * log_likelihood + n_params * np.log(n_samples)
    
    def aic(self, X):
        """
        Akaike Information Criterion for the model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to evaluate
            
        Returns:
        --------
        aic : float
            AIC score
        """
        log_likelihood = np.sum(self.score_samples(X))
        n_params = self._n_parameters()
        return -2 * log_likelihood + 2 * n_params
    
    def _n_parameters(self):
        """Calculate number of parameters in the model."""
        n_features = self.means.shape[1]
        # Weights (n_components - 1 because they sum to 1)
        n_weights = self.n_components - 1
        # Means
        n_means = self.n_components * n_features
        # Covariances
        if self.covariance_type == 'full':
            n_covariances = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            n_covariances = self.n_components * n_features
        elif self.covariance_type == 'spherical':
            n_covariances = self.n_components
        else:  # tied
            n_covariances = n_features * (n_features + 1) / 2
        
        return int(n_weights + n_means + n_covariances)


def find_optimal_components(X, max_components=10, criterion='bic'):
    """
    Find optimal number of components using information criteria.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    max_components : int, default=10
        Maximum number of components to test
    criterion : str, default='bic'
        Criterion to use ('bic' or 'aic')
        
    Returns:
    --------
    results : dict
        Dictionary containing scores for each number of components
    """
    scores = []
    models = []
    
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        
        if criterion == 'bic':
            score = gmm.bic(X)
        else:  # aic
            score = gmm.aic(X)
        
        scores.append(score)
        models.append(gmm)
    
    optimal_n = np.argmin(scores) + 1
    
    return {
        'n_components': list(range(1, max_components + 1)),
        'scores': scores,
        'optimal_n_components': optimal_n,
        'models': models
    }


def plot_gmm_results(X, gmm, figsize=(12, 5)):
    """
    Plot GMM results including data points and component distributions.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data points
    gmm : GaussianMixture
        Fitted GMM model
    figsize : tuple, default=(12, 5)
        Figure size
    """
    if X.shape[1] != 2:
        print("Plotting only works for 2D data")
        return
    
    plt.figure(figsize=figsize)
    
    # Plot data points colored by predicted cluster
    labels = gmm.predict(X)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    
    # Plot means
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], 
                c='red', marker='x', s=200, linewidths=3, label='Component Means')
    
    plt.title('Gaussian Mixture Model Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_gmm_implementations(X, n_components=3):
    """
    Compare custom GMM implementation with scikit-learn's implementation.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to cluster
    n_components : int, default=3
        Number of components
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both implementations
    """
    # Custom implementation
    custom_gmm = GaussianMixture(n_components=n_components, random_state=42)
    custom_gmm.fit(X)
    custom_labels = custom_gmm.predict(X)
    custom_log_likelihood = np.sum(custom_gmm.score_samples(X))
    
    # Scikit-learn implementation
    sklearn_gmm = SklearnGMM(n_components=n_components, random_state=42)
    sklearn_gmm.fit(X)
    sklearn_labels = sklearn_gmm.predict(X)
    sklearn_log_likelihood = sklearn_gmm.score(X) * X.shape[0]
    
    results = {
        'custom': {
            'labels': custom_labels,
            'log_likelihood': custom_log_likelihood,
            'means': custom_gmm.means,
            'weights': custom_gmm.weights
        },
        'sklearn': {
            'labels': sklearn_labels,
            'log_likelihood': sklearn_log_likelihood,
            'means': sklearn_gmm.means_,
            'weights': sklearn_gmm.weights_
        }
    }
    
    # Calculate agreement if possible
    try:
        ari = adjusted_rand_score(custom_labels, sklearn_labels)
        results['adjusted_rand_score'] = ari
    except:
        results['adjusted_rand_score'] = None
    
    return results


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    
    # Visualize results
    if X_scaled.shape[1] == 2:
        plot_gmm_results(X_scaled, gmm)
    
    # Print model information
    print("Gaussian Mixture Model Results:")
    print(f"Converged: {gmm.converged}")
    print(f"Iterations: {gmm.n_iter}")
    print(f"Component weights: {gmm.weights}")
    print(f"BIC: {gmm.bic(X_scaled):.2f}")
    print(f"AIC: {gmm.aic(X_scaled):.2f}")
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Find optimal number of components
    print("\nFinding optimal number of components...")
    bic_results = find_optimal_components(X_scaled, max_components=8, criterion='bic')
    aic_results = find_optimal_components(X_scaled, max_components=8, criterion='aic')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bic_results['n_components'], bic_results['scores'], 'bo-')
    plt.axvline(bic_results['optimal_n_components'], color='red', linestyle='--', 
                label=f'Optimal BIC: {bic_results["optimal_n_components"]}')
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.title('BIC vs Number of Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(aic_results['n_components'], aic_results['scores'], 'ro-')
    plt.axvline(aic_results['optimal_n_components'], color='blue', linestyle='--', 
                label=f'Optimal AIC: {aic_results["optimal_n_components"]}')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC Score')
    plt.title('AIC vs Number of Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Optimal number of components (BIC): {bic_results['optimal_n_components']}")
    print(f"Optimal number of components (AIC): {aic_results['optimal_n_components']}")
    
    # Compare implementations
    print("\nComparing implementations...")
    comparison_results = compare_gmm_implementations(X_scaled, n_components=3)
    print(f"Custom implementation log-likelihood: {comparison_results['custom']['log_likelihood']:.2f}")
    print(f"Scikit-learn implementation log-likelihood: {comparison_results['sklearn']['log_likelihood']:.2f}")
    
    if 'adjusted_rand_score' in comparison_results:
        print(f"Agreement between implementations: {comparison_results['adjusted_rand_score']:.4f}")