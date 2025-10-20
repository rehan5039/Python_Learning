"""
Neural Network Activation Functions

This module demonstrates various activation functions used in neural networks,
their properties, and when to use each one.
"""

import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    """Base class for activation functions"""
    
    def __init__(self, name):
        self.name = name
    
    def forward(self, x):
        """Forward pass - compute activation"""
        raise NotImplementedError
    
    def derivative(self, x):
        """Derivative of the activation function"""
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    def __init__(self):
        super().__init__("Sigmoid")
    
    def forward(self, x):
        """Sigmoid function: 1 / (1 + e^(-x))"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
        s = self.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function"""
    
    def __init__(self):
        super().__init__("Tanh")
    
    def forward(self, x):
        """Tanh function: (e^x - e^(-x)) / (e^x + e^(-x))"""
        return np.tanh(x)
    
    def derivative(self, x):
        """Derivative of tanh: 1 - tanh(x)^2"""
        t = self.forward(x)
        return 1 - np.power(t, 2)

class ReLU(ActivationFunction):
    """Rectified Linear Unit activation function"""
    
    def __init__(self):
        super().__init__("ReLU")
    
    def forward(self, x):
        """ReLU function: max(0, x)"""
        return np.maximum(0, x)
    
    def derivative(self, x):
        """Derivative of ReLU: 1 if x > 0 else 0"""
        return (x > 0).astype(float)

class LeakyReLU(ActivationFunction):
    """Leaky ReLU activation function"""
    
    def __init__(self, alpha=0.01):
        super().__init__("LeakyReLU")
        self.alpha = alpha
    
    def forward(self, x):
        """Leaky ReLU: max(alpha*x, x)"""
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, self.alpha)

class Softmax(ActivationFunction):
    """Softmax activation function"""
    
    def __init__(self):
        super().__init__("Softmax")
    
    def forward(self, x):
        """Softmax function: e^x_i / sum(e^x_j)"""
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x):
        """Derivative of softmax (Jacobian matrix)"""
        # For simplicity, we'll return a simplified version
        # In practice, this would be a Jacobian matrix
        s = self.forward(x)
        return s * (1 - s)

def plot_activation_functions():
    """Plot various activation functions"""
    x = np.linspace(-5, 5, 1000)
    
    # Create activation function instances
    activations = [
        Sigmoid(),
        Tanh(),
        ReLU(),
        LeakyReLU()
    ]
    
    # Plot activation functions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, activation in enumerate(activations):
        y = activation.forward(x)
        axes[i].plot(x, y, linewidth=2)
        axes[i].set_title(f'{activation.name} Activation Function')
        axes[i].set_xlabel('Input')
        axes[i].set_ylabel('Output')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_activations():
    """Compare different activation functions"""
    print("Activation Function Comparison:")
    print("=" * 40)
    
    # Test input values
    test_values = np.array([-2, -1, 0, 1, 2])
    
    # Activation functions to compare
    activations = [
        Sigmoid(),
        Tanh(),
        ReLU(),
        LeakyReLU()
    ]
    
    # Print comparison table
    print(f"{'Input':<8}", end="")
    for activation in activations:
        print(f"{activation.name:<12}", end="")
    print()
    
    for x in test_values:
        print(f"{x:<8.1f}", end="")
        for activation in activations:
            result = activation.forward(np.array([x]))[0]
            print(f"{result:<12.3f}", end="")
        print()

# Example usage and demonstration
if __name__ == "__main__":
    # Compare activation functions
    compare_activations()
    
    # Show softmax example
    print("\nSoftmax Example:")
    print("=" * 20)
    softmax = Softmax()
    logits = np.array([2.0, 1.0, 0.1])
    probabilities = softmax.forward(logits)
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    print(f"Sum of probabilities: {np.sum(probabilities):.3f}")
    
    # Plot activation functions
    print("\nGenerating plots of activation functions...")
    plot_activation_functions()