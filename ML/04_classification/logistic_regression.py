"""
Logistic Regression Implementation

This module covers the implementation of Logistic Regression for binary and multiclass classification:
- Binary logistic regression
- Multiclass logistic regression (one-vs-rest, softmax)
- Gradient descent optimization
- Regularization techniques
- Model evaluation and interpretation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    """
    Logistic Regression classifier implementation from scratch.
    
    Supports both binary and multiclass classification using:
    - Binary classification with sigmoid activation
    - Multiclass with softmax activation (one-vs-rest or multinomial)
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000, 
                 regularization: str = None, reg_param: float = 0.01):
        """
        Initialize Logistic Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations for convergence
            regularization: Type of regularization ('l1', 'l2', or None)
            reg_param: Regularization parameter (lambda)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.reg_param = reg_param
        self.weights = None
        self.bias = None
        self.costs = []
        self.classes = None
        self.is_multiclass = False
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function for binary classification.
        
        Args:
            z: Input array
            
        Returns:
            Sigmoid transformed values
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax activation function for multiclass classification.
        
        Args:
            z: Input array (n_samples, n_classes)
            
        Returns:
            Softmax probabilities (n_samples, n_classes)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _binary_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy cost.
        
        Args:
            y: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Cost value
        """
        m = len(y)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.reg_param / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.reg_param / m) * np.sum(np.abs(self.weights))
            
        return cost
    
    def _multiclass_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute multiclass cross-entropy cost.
        
        Args:
            y: True labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Cost value
        """
        m = len(y)
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(np.sum(y * np.log(y_pred), axis=1))
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.reg_param / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.reg_param / m) * np.sum(np.abs(self.weights))
            
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        # Get unique classes
        self.classes = np.unique(y)
        self.is_multiclass = len(self.classes) > 2
        
        # Initialize parameters
        n_samples, n_features = X.shape
        
        if not self.is_multiclass:
            # Binary classification
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0
            
            # Convert labels to binary (0, 1)
            y_binary = (y == self.classes[1]).astype(int)
            
            # Gradient descent
            for i in range(self.max_iterations):
                # Forward pass
                z = np.dot(X, self.weights) + self.bias
                y_pred = self._sigmoid(z)
                
                # Compute cost
                cost = self._binary_cost(y_binary, y_pred)
                self.costs.append(cost)
                
                # Backward pass (gradients)
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_binary))
                db = (1 / n_samples) * np.sum(y_pred - y_binary)
                
                # Add regularization terms
                if self.regularization == 'l2':
                    dw += (self.reg_param / n_samples) * self.weights
                elif self.regularization == 'l1':
                    dw += (self.reg_param / n_samples) * np.sign(self.weights)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Early stopping
                if i > 0 and abs(self.costs[-2] - self.costs[-1]) < 1e-8:
                    break
                    
        else:
            # Multiclass classification using one-vs-rest
            n_classes = len(self.classes)
            self.weights = np.random.normal(0, 0.01, (n_features, n_classes))
            self.bias = np.zeros(n_classes)
            
            # Convert to one-hot encoding
            y_onehot = np.eye(n_classes)[y]
            
            # Gradient descent
            for i in range(self.max_iterations):
                # Forward pass
                z = np.dot(X, self.weights) + self.bias
                y_pred = self._softmax(z)
                
                # Compute cost
                cost = self._multiclass_cost(y_onehot, y_pred)
                self.costs.append(cost)
                
                # Backward pass (gradients)
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_onehot))
                db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
                
                # Add regularization terms
                if self.regularization == 'l2':
                    dw += (self.reg_param / n_samples) * self.weights
                elif self.regularization == 'l1':
                    dw += (self.reg_param / n_samples) * np.sign(self.weights)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Early stopping
                if i > 0 and abs(self.costs[-2] - self.costs[-1]) < 1e-8:
                    break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        if not self.is_multiclass:
            z = np.dot(X, self.weights) + self.bias
            return self._sigmoid(z)
        else:
            z = np.dot(X, self.weights) + self.bias
            return self._softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.is_multiclass:
            probabilities = self.predict_proba(X)
            return self.classes[(probabilities > 0.5).astype(int)]
        else:
            probabilities = self.predict_proba(X)
            return self.classes[np.argmax(probabilities, axis=1)]


def generate_sample_data(n_samples: int = 1000, n_features: int = 5, 
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
        n_informative=min(n_features, 3),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_pred_proba: np.ndarray = None) -> dict:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    # Add ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                 multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 'N/A'
    
    return metrics


def plot_decision_boundary(model: LogisticRegression, X: np.ndarray, y: np.ndarray):
    """
    Plot decision boundary for 2D classification problems.
    
    Args:
        model: Trained LogisticRegression model
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
    plt.title('Logistic Regression Decision Boundary')
    plt.show()


def plot_cost_history(costs: List[float]):
    """
    Plot training cost history.
    
    Args:
        costs: List of cost values during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Cost History')
    plt.grid(True)
    plt.show()


def demo():
    """Demonstrate Logistic Regression implementation."""
    print("=== Logistic Regression Demo ===\n")
    
    # Binary Classification
    print("1. Binary Classification:")
    X_binary, y_binary = generate_sample_data(n_samples=1000, n_features=2, n_classes=2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model_binary = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model_binary.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_binary = model_binary.predict(X_test_scaled)
    y_pred_proba_binary = model_binary.predict_proba(X_test_scaled)
    
    # Evaluate
    metrics_binary = evaluate_model(y_test, y_pred_binary, y_pred_proba_binary)
    print(f"   Accuracy: {metrics_binary['accuracy']:.4f}")
    print(f"   Precision: {metrics_binary['precision']:.4f}")
    print(f"   Recall: {metrics_binary['recall']:.4f}")
    print(f"   F1-Score: {metrics_binary['f1_score']:.4f}")
    
    # Plot results (for 2D data)
    plot_decision_boundary(model_binary, X_test_scaled, y_test)
    plot_cost_history(model_binary.costs)
    
    # Multiclass Classification
    print("\n2. Multiclass Classification:")
    X_multi, y_multi = generate_sample_data(n_samples=1000, n_features=5, n_classes=3)
    
    # Split data
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_multi = StandardScaler()
    X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
    X_test_multi_scaled = scaler_multi.transform(X_test_multi)
    
    # Train model
    model_multi = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    model_multi.fit(X_train_multi_scaled, y_train_multi)
    
    # Make predictions
    y_pred_multi = model_multi.predict(X_test_multi_scaled)
    y_pred_proba_multi = model_multi.predict_proba(X_test_multi_scaled)
    
    # Evaluate
    metrics_multi = evaluate_model(y_test_multi, y_pred_multi, y_pred_proba_multi)
    print(f"   Accuracy: {metrics_multi['accuracy']:.4f}")
    print(f"   Precision: {metrics_multi['precision']:.4f}")
    print(f"   Recall: {metrics_multi['recall']:.4f}")
    print(f"   F1-Score: {metrics_multi['f1_score']:.4f}")
    
    # Compare with sklearn
    print("\n3. Comparison with Scikit-learn:")
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    
    sklearn_model = SklearnLogisticRegression(random_state=42, max_iter=1000)
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_model.predict(X_test_scaled)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"   Our Implementation Accuracy: {metrics_binary['accuracy']:.4f}")
    print(f"   Scikit-learn Accuracy: {sklearn_accuracy:.4f}")


if __name__ == "__main__":
    demo()