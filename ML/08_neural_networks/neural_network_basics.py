"""
Neural Network Basics Implementation
=============================

This module provides implementations of fundamental neural network concepts
with detailed explanations and practical examples.

Topics Covered:
- Artificial neurons and perceptrons
- Multi-layer perceptrons
- Network architecture fundamentals
- Basic implementation from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class Perceptron:
    """
    Single Perceptron implementation.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for weight updates.
    max_iterations : int, default=1000
        Maximum number of training iterations.
    
    Attributes:
    -----------
    weights : array
        Learned weights for the perceptron.
    bias : float
        Bias term.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
    
    def fit(self, X, y):
        """
        Train the perceptron.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Initialize weights and bias
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1 if needed
        y = np.where(y <= 0, -1, 1)
        
        # Training loop
        for _ in range(self.max_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                # Prediction
                linear_output = np.dot(xi, self.weights) + self.bias
                prediction = np.where(linear_output >= 0, 1, -1)
                
                # Update weights if misclassified
                if prediction != yi:
                    update = self.learning_rate * yi
                    self.weights += update * xi
                    self.bias += update
                    errors += 1
            
            # Stop if no errors
            if errors == 0:
                break
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)


class MultiLayerPerceptron:
    """
    Multi-Layer Perceptron implementation.
    
    Parameters:
    -----------
    hidden_layers : list, default=[10]
        Number of neurons in each hidden layer.
    learning_rate : float, default=0.01
        Learning rate for weight updates.
    max_iterations : int, default=1000
        Maximum number of training iterations.
    activation : str, default='sigmoid'
        Activation function ('sigmoid', 'tanh', 'relu').
    
    Attributes:
    -----------
    weights : list
        List of weight matrices for each layer.
    biases : list
        List of bias vectors for each layer.
    """
    
    def __init__(self, hidden_layers=[10], learning_rate=0.01, max_iterations=1000, activation='sigmoid'):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.activation = activation
    
    def _activate(self, x):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Activation function must be 'sigmoid', 'tanh', or 'relu'")
    
    def _activate_derivative(self, x):
        """Apply derivative of activation function."""
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1 - x**2
        elif self.activation == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError("Activation function must be 'sigmoid', 'tanh', or 'relu'")
    
    def fit(self, X, y):
        """
        Train the multi-layer perceptron.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        """
        # Convert labels to one-hot encoding if needed
        if len(np.unique(y)) > 2:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            y_encoded = lb.fit_transform(y)
            self.label_binarizer_ = lb
        else:
            y_encoded = y.reshape(-1, 1)
        
        # Initialize network architecture
        layer_sizes = [X.shape[1]] + self.hidden_layers + [y_encoded.shape[1]]
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
        
        # Training loop
        self.losses = []
        for iteration in range(self.max_iterations):
            # Forward propagation
            activations = [X]
            zs = []
            
            for i in range(len(self.weights)):
                z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                zs.append(z)
                a = self._activate(z)
                activations.append(a)
            
            # Calculate loss
            loss = np.mean((activations[-1] - y_encoded) ** 2)
            self.losses.append(loss)
            
            # Backward propagation
            # Output layer error
            delta = (activations[-1] - y_encoded) * self._activate_derivative(activations[-1])
            
            # Backpropagate error
            deltas = [delta]
            for i in range(len(self.weights) - 2, -1, -1):
                delta = np.dot(deltas[0], self.weights[i+1].T) * self._activate_derivative(activations[i+1])
                deltas.insert(0, delta)
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / X.shape[0]
                self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Forward propagation
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z)
        return a
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples.
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels.
        """
        probabilities = self.predict_proba(X)
        if probabilities.shape[1] == 1:
            return (probabilities > 0.5).astype(int).flatten()
        else:
            return np.argmax(probabilities, axis=1)


def compare_perceptron_mlp():
    """
    Compare Perceptron and Multi-Layer Perceptron performance.
    """
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Perceptron
    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    perceptron.fit(X_train, y_train)
    y_pred_perceptron = perceptron.predict(X_test)
    acc_perceptron = accuracy_score(y_test, y_pred_perceptron)
    
    # Train Multi-Layer Perceptron
    mlp = MultiLayerPerceptron(hidden_layers=[10], learning_rate=0.1, max_iterations=1000)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    
    print("Perceptron vs Multi-Layer Perceptron Comparison:")
    print(f"Perceptron Accuracy: {acc_perceptron:.4f}")
    print(f"MLP Accuracy: {acc_mlp:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot data
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data')
    plt.colorbar(scatter)
    
    # Plot Perceptron results
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_perceptron, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Perceptron (Accuracy: {acc_perceptron:.3f})')
    plt.colorbar(scatter)
    
    # Plot MLP results
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_mlp, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'MLP (Accuracy: {acc_mlp:.3f})')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return perceptron, mlp


def neural_network_architecture_demo():
    """
    Demonstrate different neural network architectures.
    """
    print("=== Neural Network Architecture Overview ===")
    
    architectures = {
        "Single Layer Perceptron": {
            "Description": "One input layer connected directly to output layer",
            "Layers": ["Input", "Output"],
            "Use Cases": ["Linear classification", "Simple pattern recognition"]
        },
        "Multi-Layer Perceptron": {
            "Description": "Input layer, one or more hidden layers, output layer",
            "Layers": ["Input", "Hidden", "Output"],
            "Use Cases": ["Non-linear classification", "Function approximation"]
        },
        "Deep Neural Network": {
            "Description": "Multiple hidden layers for complex pattern recognition",
            "Layers": ["Input", "Hidden 1", "Hidden 2", "...", "Hidden N", "Output"],
            "Use Cases": ["Image recognition", "Natural language processing", "Complex data modeling"]
        }
    }
    
    for arch_name, details in architectures.items():
        print(f"\n{arch_name}:")
        print(f"  Description: {details['Description']}")
        print(f"  Layers: {' -> '.join(details['Layers'])}")
        print(f"  Use Cases: {', '.join(details['Use Cases'])}")


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate neural network architectures
    neural_network_architecture_demo()
    print("\n" + "="*60 + "\n")
    
    # Compare Perceptron and MLP
    perceptron, mlp = compare_perceptron_mlp()
    print("\n" + "="*60 + "\n")
    
    # Plot MLP training loss
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('MLP Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n=== Summary ===")
    print("This module covered:")
    print("1. Perceptron implementation and training")
    print("2. Multi-Layer Perceptron with backpropagation")
    print("3. Different activation functions")
    print("4. Network architecture fundamentals")
    print("5. Comparison of simple and complex neural networks")
    print("\nIn the full Neural Networks chapter, you'll learn:")
    print("- Advanced optimization techniques")
    print("- Regularization methods to prevent overfitting")
    print("- Practical implementation with TensorFlow/Keras")
    print("- Real-world applications and case studies")