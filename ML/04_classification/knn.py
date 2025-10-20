"""
K-Nearest Neighbors (KNN) Implementation

This module covers the implementation of K-Nearest Neighbors algorithm for classification:
- Distance metrics (Euclidean, Manhattan, Minkowski)
- K-selection strategies
- Weighted voting schemes
- Efficient search algorithms (KD-tree, Ball tree)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier implementation from scratch.
    
    Supports different distance metrics and voting schemes for robust classification.
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean', 
                 weights: str = 'uniform'):
        """
        Initialize KNN classifier.
        
        Args:
            k: Number of neighbors to consider
            distance_metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
            weights: Weighting scheme ('uniform' or 'distance')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes = None
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            Manhattan distance
        """
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray, p: int = 3) -> float:
        """
        Calculate Minkowski distance between two points.
        
        Args:
            x1: First point
            x2: Second point
            p: Order of Minkowski distance
            
        Returns:
            Minkowski distance
        """
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate distance between two points using specified metric.
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            Distance between points
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.distance_metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNearestNeighbors':
        """
        Store training data for KNN classification.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes = np.unique(y)
        return self
    
    def _get_neighbors(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a given point.
        
        Args:
            x: Query point (n_features,)
            
        Returns:
            Tuple of (neighbor_indices, distances)
        """
        # Calculate distances to all training points
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._calculate_distance(x, x_train)
            distances.append((i, dist))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:self.k]
        
        # Extract indices and distances
        indices = np.array([idx for idx, _ in k_nearest])
        dists = np.array([dist for _, dist in k_nearest])
        
        return indices, dists
    
    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict class for a single point.
        
        Args:
            x: Query point (n_features,)
            
        Returns:
            Predicted class label
        """
        # Get k nearest neighbors
        neighbor_indices, distances = self._get_neighbors(x)
        
        # Get neighbor labels
        neighbor_labels = self.y_train[neighbor_indices]
        
        if self.weights == 'uniform':
            # Simple majority voting
            most_common = Counter(neighbor_labels).most_common(1)
            return most_common[0][0]
        else:
            # Distance-weighted voting
            # Avoid division by zero for identical points
            weights = 1 / (distances + 1e-8)
            
            # Weighted voting
            weighted_votes = {}
            for label, weight in zip(neighbor_labels, weights):
                if label in weighted_votes:
                    weighted_votes[label] += weight
                else:
                    weighted_votes[label] = weight
            
            # Return class with highest weighted votes
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple points.
        
        Args:
            X: Query points (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)


def generate_sample_data(n_samples: int = 1000, n_features: int = 2, 
                        n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y) features and labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features, 2),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


def find_optimal_k(X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, 
                  k_range: range = range(1, 21)) -> int:
    """
    Find optimal k value using validation set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        k_range: Range of k values to test
        
    Returns:
        Optimal k value
    """
    best_k = 1
    best_accuracy = 0
    
    for k in k_range:
        knn = KNearestNeighbors(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    return best_k


def plot_knn_decision_boundary(model: KNearestNeighbors, X: np.ndarray, y: np.ndarray):
    """
    Plot decision boundary for 2D KNN classification.
    
    Args:
        model: Trained KNN model
        X: Features (n_samples, 2)
        y: Labels (n_samples,)
    """
    if X.shape[1] != 2:
        print("Decision boundary plotting only supported for 2D features")
        return
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'KNN Decision Boundary (k={model.k})')
    plt.show()


def compare_distance_metrics(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray):
    """
    Compare performance of different distance metrics.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    metrics = ['euclidean', 'manhattan', 'minkowski']
    results = {}
    
    for metric in metrics:
        knn = KNearestNeighbors(k=5, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[metric] = accuracy
    
    print("Distance Metric Comparison:")
    for metric, accuracy in results.items():
        print(f"  {metric.capitalize()}: {accuracy:.4f}")


def demo():
    """Demonstrate KNN implementation."""
    print("=== K-Nearest Neighbors Demo ===\n")
    
    # Generate sample data
    print("1. Generating Sample Data:")
    X, y = generate_sample_data(n_samples=500, n_features=2, n_classes=3)
    print(f"   Data shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal k
    print("\n2. Finding Optimal K:")
    optimal_k = find_optimal_k(X_train_scaled, y_train, X_val_scaled, y_val)
    print(f"   Optimal k: {optimal_k}")
    
    # Train KNN with optimal k
    print("\n3. Training KNN Model:")
    knn = KNearestNeighbors(k=optimal_k)
    knn.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot decision boundary
    plot_knn_decision_boundary(knn, X_test_scaled, y_test)
    
    # Compare distance metrics
    print("\n4. Distance Metric Comparison:")
    compare_distance_metrics(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Compare with sklearn
    print("\n5. Comparison with Scikit-learn:")
    from sklearn.neighbors import KNeighborsClassifier
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    sklearn_knn.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_knn.predict(X_test_scaled)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"   Our Implementation Accuracy: {accuracy:.4f}")
    print(f"   Scikit-learn Accuracy: {sklearn_accuracy:.4f}")


if __name__ == "__main__":
    demo()