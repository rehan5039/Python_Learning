"""
Machine Learning Optimization with DSA Principles

This module demonstrates how to optimize machine learning algorithms using Data Structures and Algorithms:
- Efficient implementation of ML algorithms
- Feature selection and dimensionality reduction
- Model optimization techniques
- Hyperparameter tuning strategies
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import time


class OptimizedKMeans:
    """
    Optimized K-Means implementation using efficient data structures and algorithms.
    """
    
    def __init__(self, k: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def fit(self, X: np.ndarray) -> 'OptimizedKMeans':
        """
        Fit K-Means clustering to data.
        
        Time Complexity: O(n * k * d * max_iters) where n samples, k clusters, d dimensions
        Space Complexity: O(n + k * d)
        """
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for iteration in range(self.max_iters):
            # Assign points to closest centroids
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
            
            # Check for convergence
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                break
                
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)


class OptimizedDecisionTree:
    """
    Optimized Decision Tree implementation using efficient splitting algorithms.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find the best split using efficient algorithms."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # For each feature, find best threshold
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # For each threshold, calculate Gini impurity
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                weighted_gini = (np.sum(left_mask) * left_gini + 
                               np.sum(right_mask) * right_gini) / len(y)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict:
        """Recursively build decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        
        if feature_idx is None:
            leaf_value = np.bincount(y).argmax()
            return {'leaf': True, 'value': leaf_value}
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimizedDecisionTree':
        """Fit decision tree to training data."""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, tree: Dict) -> int:
        """Predict class for a single sample."""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature_idx']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for samples."""
        return np.array([self._predict_sample(x, self.tree) for x in X])


def feature_selection_optimization(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Optimize feature selection using correlation and variance analysis.
    
    Time Complexity: O(n * m^2) where n samples, m features
    Space Complexity: O(m)
    """
    n_samples, n_features = X.shape
    
    # Calculate variance for each feature
    variances = np.var(X, axis=0)
    
    # Calculate correlation with target
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
    correlations = np.abs(correlations)  # Take absolute values
    
    # Combine variance and correlation (simple approach)
    scores = variances * correlations
    
    # Select top k features
    top_features = np.argsort(scores)[-k:]
    
    return top_features


def gradient_descent_optimization(X: np.ndarray, y: np.ndarray, 
                                learning_rate: float = 0.01, 
                                max_iters: int = 1000,
                                tol: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
    """
    Optimize linear regression using gradient descent.
    
    Time Complexity: O(n * d * max_iters) where n samples, d features
    Space Complexity: O(d)
    """
    n_samples, n_features = X.shape
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    n_features += 1
    
    # Initialize weights
    weights = np.random.randn(n_features) * 0.01
    costs = []
    
    for iteration in range(max_iters):
        # Forward pass
        predictions = X_with_bias.dot(weights)
        errors = predictions - y
        
        # Calculate cost
        cost = np.mean(errors ** 2) / 2
        costs.append(cost)
        
        # Backward pass (gradient calculation)
        gradients = X_with_bias.T.dot(errors) / n_samples
        
        # Update weights
        weights -= learning_rate * gradients
        
        # Check for convergence
        if iteration > 0 and abs(costs[-2] - costs[-1]) < tol:
            break
    
    return weights, costs


def cross_validation_optimization(X: np.ndarray, y: np.ndarray, 
                                k_folds: int = 5) -> Dict[str, float]:
    """
    Optimize model evaluation using k-fold cross-validation.
    
    Time Complexity: O(k * n * d) where k folds, n samples, d features
    Space Complexity: O(n)
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k_folds
    
    scores = []
    
    for fold in range(k_folds):
        # Split data
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size
        
        X_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]
        
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Train model (simple example with linear regression)
        X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
        weights = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]
        
        # Evaluate model
        X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
        predictions = X_test_bias.dot(weights)
        mse = np.mean((predictions - y_test) ** 2)
        scores.append(mse)
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores)
    }


def performance_comparison():
    """Compare performance of different ML optimization techniques."""
    print("=== ML Optimization Performance Comparison ===\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples) * 0.1
    
    # Test K-Means optimization
    print("1. K-Means Clustering:")
    start_time = time.time()
    kmeans = OptimizedKMeans(k=3)
    kmeans.fit(X)
    kmeans_time = time.time() - start_time
    print(f"   K-Means time: {kmeans_time:.6f} seconds")
    print(f"   Number of clusters: {kmeans.k}")
    
    # Test Decision Tree optimization
    print("\n2. Decision Tree:")
    y_classification = (y > np.median(y)).astype(int)  # Convert to classification
    start_time = time.time()
    dt = OptimizedDecisionTree(max_depth=5)
    dt.fit(X[:500], y_classification[:500])  # Use subset for faster training
    dt_time = time.time() - start_time
    print(f"   Decision Tree time: {dt_time:.6f} seconds")
    
    # Test predictions
    start_time = time.time()
    predictions = dt.predict(X[500:600])
    pred_time = time.time() - start_time
    print(f"   Prediction time (100 samples): {pred_time:.6f} seconds")
    
    # Test feature selection
    print("\n3. Feature Selection:")
    start_time = time.time()
    selected_features = feature_selection_optimization(X, y, k=5)
    feature_time = time.time() - start_time
    print(f"   Feature selection time: {feature_time:.6f} seconds")
    print(f"   Selected features: {selected_features}")
    
    # Test gradient descent
    print("\n4. Gradient Descent:")
    start_time = time.time()
    weights, costs = gradient_descent_optimization(X, y)
    gd_time = time.time() - start_time
    print(f"   Gradient descent time: {gd_time:.6f} seconds")
    print(f"   Final cost: {costs[-1]:.6f}")
    print(f"   Number of iterations: {len(costs)}")
    
    # Test cross-validation
    print("\n5. Cross-Validation:")
    start_time = time.time()
    cv_scores = cross_validation_optimization(X, y)
    cv_time = time.time() - start_time
    print(f"   Cross-validation time: {cv_time:.6f} seconds")
    print(f"   Mean CV score: {cv_scores['mean_score']:.6f}")


def demo():
    """Demonstrate ML optimization techniques."""
    print("=== ML Optimization with DSA ===\n")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y_regression = X.dot(np.array([1, -2, 0.5, 3, -1])) + np.random.randn(200) * 0.1
    y_classification = (y_regression > np.median(y_regression)).astype(int)
    
    print("Sample data created:")
    print(f"  Features shape: {X.shape}")
    print(f"  Regression target shape: {y_regression.shape}")
    print(f"  Classification target shape: {y_classification.shape}")
    
    # Test K-Means
    print("\n1. K-Means Clustering:")
    kmeans = OptimizedKMeans(k=3)
    kmeans.fit(X)
    print(f"  Centroids shape: {kmeans.centroids.shape}")
    print(f"  Labels for first 10 samples: {kmeans.labels[:10]}")
    
    # Test Decision Tree
    print("\n2. Decision Tree:")
    dt = OptimizedDecisionTree(max_depth=3)
    dt.fit(X[:100], y_classification[:100])
    predictions = dt.predict(X[100:110])
    print(f"  Predictions for first 10 test samples: {predictions}")
    
    # Test feature selection
    print("\n3. Feature Selection:")
    selected_features = feature_selection_optimization(X, y_regression, k=3)
    print(f"  Top 3 features: {selected_features}")
    
    # Test gradient descent
    print("\n4. Gradient Descent:")
    weights, costs = gradient_descent_optimization(X, y_regression)
    print(f"  Weights shape: {weights.shape}")
    print(f"  Final cost: {costs[-1]:.6f}")
    
    # Test cross-validation
    print("\n5. Cross-Validation:")
    cv_scores = cross_validation_optimization(X, y_regression)
    print(f"  Cross-validation scores: {cv_scores}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()