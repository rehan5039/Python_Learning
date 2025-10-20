"""
Neural Network Architectures

This module demonstrates different neural network architectures,
from simple perceptrons to more complex multi-layer networks.
"""

import numpy as np
from .activation_functions import ReLU, Sigmoid, Tanh

class NeuralNetwork:
    """Base class for neural networks"""
    
    def __init__(self, layers, activations):
        """
        Initialize neural network
        
        Args:
            layers (list): List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations (list): List of activation functions for each layer
        """
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier initialization"""
        for i in range(len(self.layers) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (self.layers[i] + self.layers[i+1]))
            weight = np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1]))
            bias = np.zeros((1, self.layers[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)
    
    def forward(self, X):
        """
        Forward propagation through the network
        
        Args:
            X (np.array): Input data of shape (n_samples, n_features)
            
        Returns:
            np.array: Output predictions
        """
        activations = [X]
        current_input = X
        
        # Forward pass through each layer
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            
            # Apply activation function
            if isinstance(self.activations[i], str):
                activation_func = self._get_activation_function(self.activations[i])
            else:
                activation_func = self.activations[i]
            
            a = activation_func.forward(z)
            activations.append(a)
            current_input = a
        
        return current_input, activations
    
    def _get_activation_function(self, name):
        """Get activation function by name"""
        activation_map = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'tanh': Tanh()
        }
        return activation_map.get(name.lower(), ReLU())
    
    def predict(self, X):
        """
        Make predictions on input data
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predictions
        """
        output, _ = self.forward(X)
        return output

class Perceptron(NeuralNetwork):
    """Single-layer perceptron"""
    
    def __init__(self, input_size, output_size=1):
        """
        Initialize perceptron
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output units
        """
        super().__init__(
            layers=[input_size, output_size],
            activations=['sigmoid']
        )

class MultiLayerPerceptron(NeuralNetwork):
    """Multi-layer perceptron (feedforward neural network)"""
    
    def __init__(self, input_size, hidden_layers, output_size, activations=None):
        """
        Initialize MLP
        
        Args:
            input_size (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            output_size (int): Number of output units
            activations (list): List of activation functions for each layer
        """
        # Define layer sizes
        layers = [input_size] + hidden_layers + [output_size]
        
        # Default activations
        if activations is None:
            activations = ['relu'] * (len(layers) - 2) + ['sigmoid']
        
        super().__init__(layers=layers, activations=activations)

class SimpleCNN:
    """Simple Convolutional Neural Network concepts"""
    
    def __init__(self, input_shape, num_filters=32, filter_size=3):
        """
        Initialize simple CNN
        
        Args:
            input_shape (tuple): Shape of input (height, width, channels)
            num_filters (int): Number of convolutional filters
            filter_size (int): Size of convolutional filters
        """
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size)
    
    def convolve(self, input_data):
        """
        Simple convolution operation
        
        Args:
            input_data (np.array): Input data
            
        Returns:
            np.array: Convolved features
        """
        height, width = input_data.shape
        output_height = height - self.filter_size + 1
        output_width = width - self.filter_size + 1
        output = np.zeros((self.num_filters, output_height, output_width))
        
        for f in range(self.num_filters):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_data[i:i+self.filter_size, j:j+self.filter_size]
                    output[f, i, j] = np.sum(region * self.filters[f])
        
        return output

def demonstrate_networks():
    """Demonstrate different network architectures"""
    print("Neural Network Architectures Demonstration:")
    print("=" * 45)
    
    # Sample data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])  # XOR problem
    
    print("Sample Data (XOR problem):")
    print("Input:")
    print(X)
    print("Expected Output:")
    print(y)
    
    # Perceptron
    print("\n1. Perceptron:")
    perceptron = Perceptron(input_size=2, output_size=1)
    pred, _ = perceptron.forward(X)
    print("Predictions:")
    print(pred)
    
    # Multi-layer Perceptron
    print("\n2. Multi-layer Perceptron:")
    mlp = MultiLayerPerceptron(
        input_size=2,
        hidden_layers=[4, 4],
        output_size=1,
        activations=['relu', 'relu', 'sigmoid']
    )
    pred, _ = mlp.forward(X)
    print("Predictions:")
    print(pred)
    
    # Simple CNN example
    print("\n3. Simple CNN Concept:")
    input_image = np.random.randn(8, 8)  # 8x8 grayscale image
    cnn = SimpleCNN(input_shape=(8, 8, 1), num_filters=2, filter_size=3)
    features = cnn.convolve(input_image)
    print(f"Input image shape: {input_image.shape}")
    print(f"Feature maps shape: {features.shape}")

# Example usage and demonstration
if __name__ == "__main__":
    demonstrate_networks()