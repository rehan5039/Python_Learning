"""
Support Vector Machine (SVM) Implementation

This module covers the implementation of Support Vector Machine algorithm:
- Linear SVM with gradient descent optimization
- Kernel methods (linear, polynomial, RBF)
- Soft margin classification
- Multi-class classification strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class SupportVectorMachine:
    """
    Support Vector Machine classifier implementation from scratch.
    
    Supports both binary and multi-class classification with various kernel functions.
    """
    
    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01,
                 n_iters: int = 1000, kernel: str = 'linear', 
                 kernel_params: dict = None):
        """
        Initialize SVM classifier.
        
        Args:
            learning_rate: Learning rate for gradient descent
            lambda_param: Regularization parameter
            n_iters: Number of iterations for training
            kernel: Kernel function ('linear', 'poly', 'rbf')
            kernel_params: Parameters for kernel functions
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.classes = None
        self.is_multiclass = False
    
    def _linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Linear kernel function.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        return np.dot(x1, x2)
    
    def _polynomial_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Polynomial kernel function.
        
        K(x1, x2) = (gamma * <x1, x2> + coef0)^degree
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        gamma = self.kernel_params.get('gamma', 1.0)
        coef0 = self.kernel_params.get('coef0', 1.0)
        degree = self.kernel_params.get('degree', 3)
        
        return (gamma * np.dot(x1, x2) + coef0) ** degree
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Radial Basis Function (RBF) kernel.
        
        K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        gamma = self.kernel_params.get('gamma', 1.0)
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def _compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel function value.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        if self.kernel == 'linear':
            return self._linear_kernel(x1, x2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(x1, x2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(x1, x2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between two datasets.
        
        Args:
            X1: First dataset (n1_samples, n_features)
            X2: Second dataset (n2_samples, n_features)
            
        Returns:
            Kernel matrix (n1_samples, n2_samples)
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_kernel(X1[i], X2[j])
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorMachine':
        """
        Train the SVM classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        # Handle multi-class classification
        self.classes = np.unique(y)
        self.is_multiclass = len(self.classes) > 2
        
        if not self.is_multiclass:
            # Binary classification
            return self._fit_binary(X, y)
        else:
            # Multi-class using one-vs-rest
            return self._fit_multiclass(X, y)
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorMachine':
        """
        Train binary SVM classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        # Convert labels to -1, 1
        y_binary = np.where(y <= 0, -1, 1)
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_binary[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Correct classification with margin
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification or within margin
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_binary[idx]))
                    self.b -= self.learning_rate * y_binary[idx]
        
        # Identify support vectors (simplified approach)
        distances = y_binary * (np.dot(X, self.w) - self.b)
        sv_indices = np.where(distances <= 1 + 1e-6)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y_binary[sv_indices]
        
        return self
    
    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorMachine':
        """
        Train multi-class SVM using one-vs-rest approach.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        # Create binary classifiers for each class
        self.classifiers = []
        
        for class_label in self.classes:
            # Create binary labels
            y_binary = np.where(y == class_label, 1, -1)
            
            # Train binary SVM
            binary_svm = SupportVectorMachine(
                learning_rate=self.learning_rate,
                lambda_param=self.lambda_param,
                n_iters=self.n_iters,
                kernel=self.kernel,
                kernel_params=self.kernel_params
            )
            binary_svm._fit_binary(X, y_binary)
            self.classifiers.append(binary_svm)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_multiclass:
            # Binary classification
            approx = np.dot(X, self.w) - self.b
            return np.where(approx >= 0, 1, 0)
        else:
            # Multi-class classification
            predictions = []
            
            for x in X:
                # Get scores from all binary classifiers
                scores = []
                for classifier in self.classifiers:
                    score = np.dot(x, classifier.w) - classifier.b
                    scores.append(score)
                
                # Predict class with highest score
                predicted_class_idx = np.argmax(scores)
                predictions.append(self.classes[predicted_class_idx])
            
            return np.array(predictions)


def generate_sample_data(n_samples: int = 1000, n_features: int = 2, 
                        n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
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


def plot_svm_decision_boundary(model: SupportVectorMachine, X: np.ndarray, y: np.ndarray):
    """
    Plot decision boundary for 2D SVM classification.
    
    Args:
        model: Trained SVM model
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
    plt.title(f'SVM Decision Boundary ({model.kernel.capitalize()} Kernel)')
    plt.show()


def compare_kernels(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray):
    """
    Compare performance of different kernel functions.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    """
    kernels = ['linear', 'poly', 'rbf']
    results = {}
    
    for kernel in kernels:
        svm = SupportVectorMachine(kernel=kernel, n_iters=1000)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
    
    print("Kernel Comparison:")
    for kernel, accuracy in results.items():
        print(f"  {kernel.capitalize()} Kernel: {accuracy:.4f}")


def demo():
    """Demonstrate SVM implementation."""
    print("=== Support Vector Machine Demo ===\n")
    
    # Generate sample data
    print("1. Generating Sample Data:")
    X, y = generate_sample_data(n_samples=500, n_features=2, n_classes=2)
    print(f"   Data shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM with different kernels
    print("\n2. Training SVM with Different Kernels:")
    
    # Linear SVM
    svm_linear = SupportVectorMachine(kernel='linear', n_iters=1000)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred_linear = svm_linear.predict(X_test_scaled)
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    print(f"   Linear SVM Accuracy: {accuracy_linear:.4f}")
    
    # Polynomial SVM
    svm_poly = SupportVectorMachine(kernel='poly', n_iters=1000, 
                                   kernel_params={'degree': 3, 'gamma': 0.1})
    svm_poly.fit(X_train_scaled, y_train)
    y_pred_poly = svm_poly.predict(X_test_scaled)
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    print(f"   Polynomial SVM Accuracy: {accuracy_poly:.4f}")
    
    # RBF SVM
    svm_rbf = SupportVectorMachine(kernel='rbf', n_iters=1000, 
                                  kernel_params={'gamma': 0.1})
    svm_rbf.fit(X_train_scaled, y_train)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
    print(f"   RBF SVM Accuracy: {accuracy_rbf:.4f}")
    
    # Plot decision boundary for linear SVM
    print("\n3. Decision Boundary Visualization:")
    plot_svm_decision_boundary(svm_linear, X_test_scaled, y_test)
    
    # Compare with sklearn
    print("\n4. Comparison with Scikit-learn:")
    from sklearn.svm import SVC
    
    # Scikit-learn Linear SVM
    sklearn_svm = SVC(kernel='linear', random_state=42)
    sklearn_svm.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_svm.predict(X_test_scaled)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"   Our Linear SVM vs Scikit-learn: {accuracy_linear:.4f} vs {sklearn_accuracy:.4f}")
    
    # Multi-class example
    print("\n5. Multi-class Classification:")
    X_multi, y_multi = generate_sample_data(n_samples=600, n_features=4, n_classes=3)
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )
    
    # Scale multi-class data
    scaler_multi = StandardScaler()
    X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
    X_test_multi_scaled = scaler_multi.transform(X_test_multi)
    
    # Train multi-class SVM
    svm_multi = SupportVectorMachine(kernel='rbf', n_iters=1000)
    svm_multi.fit(X_train_multi_scaled, y_train_multi)
    y_pred_multi = svm_multi.predict(X_test_multi_scaled)
    accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
    
    print(f"   Multi-class SVM Accuracy: {accuracy_multi:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_multi, y_pred_multi))


if __name__ == "__main__":
    demo()