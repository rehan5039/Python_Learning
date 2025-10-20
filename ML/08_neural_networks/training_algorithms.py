"""
Neural Network Training Algorithms

This module demonstrates various training algorithms used for neural networks,
including backpropagation and gradient descent variants.
"""

import numpy as np
from .network_architectures import NeuralNetwork
from .activation_functions import ReLU, Sigmoid

class GradientDescentOptimizer:
    """Basic Gradient Descent Optimizer"""
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize optimizer
        
        Args:
            learning_rate (float): Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update_parameters(self, weights, biases, dW, db):
        """
        Update parameters using gradient descent
        
        Args:
            weights (list): Current weights
            biases (list): Current biases
            dW (list): Weight gradients
            db (list): Bias gradients
            
        Returns:
            tuple: Updated weights and biases
        """
        updated_weights = []
        updated_biases = []
        
        for i in range(len(weights)):
            w = weights[i] - self.learning_rate * dW[i]
            b = biases[i] - self.learning_rate * db[i]
            updated_weights.append(w)
            updated_biases.append(b)
        
        return updated_weights, updated_biases

class AdamOptimizer:
    """Adam Optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate (float): Learning rate
            beta1 (float): Exponential decay rate for first moment
            beta2 (float): Exponential decay rate for second moment
            epsilon (float): Small constant to prevent division by zero
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None  # First moment for weights
        self.v_w = None  # Second moment for weights
        self.m_b = None  # First moment for biases
        self.v_b = None  # Second moment for biases
        self.t = 0       # Time step
    
    def update_parameters(self, weights, biases, dW, db):
        """
        Update parameters using Adam optimizer
        
        Args:
            weights (list): Current weights
            biases (list): Current biases
            dW (list): Weight gradients
            db (list): Bias gradients
            
        Returns:
            tuple: Updated weights and biases
        """
        if self.m_w is None:
            # Initialize moments
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        updated_weights = []
        updated_biases = []
        
        for i in range(len(weights)):
            # Update moments
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
            
            # Bias correction
            m_w_corrected = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_corrected = self.v_w[i] / (1 - self.beta2 ** self.t)
            m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            w = weights[i] - self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            b = biases[i] - self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
            
            updated_weights.append(w)
            updated_biases.append(b)
        
        return updated_weights, updated_biases

class NetworkTrainer:
    """Neural Network Trainer"""
    
    def __init__(self, network, optimizer=None, loss_function='mse'):
        """
        Initialize trainer
        
        Args:
            network (NeuralNetwork): Neural network to train
            optimizer (object): Optimizer for parameter updates
            loss_function (str): Loss function ('mse' or 'binary_crossentropy')
        """
        self.network = network
        self.optimizer = optimizer if optimizer else GradientDescentOptimizer()
        self.loss_function = loss_function
        self.loss_history = []
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute loss
        
        Args:
            y_true (np.array): True labels
            y_pred (np.array): Predicted values
            
        Returns:
            float: Computed loss
        """
        if self.loss_function == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss_function == 'binary_crossentropy':
            # Clip predictions to prevent log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
    
    def compute_gradients(self, X, y, activations):
        """
        Compute gradients using backpropagation
        
        Args:
            X (np.array): Input data
            y (np.array): True labels
            activations (list): Activations from forward pass
            
        Returns:
            tuple: Weight gradients and bias gradients
        """
        m = X.shape[0]  # Number of samples
        dW = []
        db = []
        
        # Compute output layer error
        if self.loss_function == 'mse':
            dZ = activations[-1] - y
        elif self.loss_function == 'binary_crossentropy':
            dZ = activations[-1] - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.network.weights))):
            # Compute gradients
            dW_i = np.dot(activations[i].T, dZ) / m
            db_i = np.sum(dZ, axis=0, keepdims=True) / m
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:  # Not the first layer
                # Compute error for previous layer
                dA = np.dot(dZ, self.network.weights[i].T)
                
                # Apply derivative of activation function
                if isinstance(self.network.activations[i-1], str):
                    activation_func = self.network._get_activation_function(self.network.activations[i-1])
                else:
                    activation_func = self.network.activations[i-1]
                
                dZ = dA * activation_func.derivative(activations[i])

        return dW, db
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X (np.array): Training data
            y (np.array): Training labels
            epochs (int): Number of training epochs
            verbose (bool): Whether to print training progress
        """
        for epoch in range(epochs):
            # Forward pass
            output, activations = self.network.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            self.loss_history.append(loss)
            
            # Compute gradients
            dW, db = self.compute_gradients(X, y, activations)
            
            # Update parameters
            self.network.weights, self.network.biases = self.optimizer.update_parameters(
                self.network.weights, self.network.biases, dW, db
            )
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def plot_loss(self):
        """Plot training loss"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.show()

def demonstrate_training():
    """Demonstrate neural network training"""
    print("Neural Network Training Demonstration:")
    print("=" * 38)
    
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    print("Training Data (XOR problem):")
    print("Inputs:")
    print(X)
    print("Targets:")
    print(y)
    
    # Create network
    network = NeuralNetwork(
        layers=[2, 4, 1],
        activations=['relu', 'sigmoid']
    )
    
    # Train with basic gradient descent
    print("\nTraining with Gradient Descent:")
    trainer_gd = NetworkTrainer(network, GradientDescentOptimizer(learning_rate=1.0))
    trainer_gd.train(X, y, epochs=1000, verbose=False)
    
    # Make predictions
    predictions, _ = network.forward(X)
    print("Final Predictions:")
    print(predictions)
    
    # Create another network for Adam optimizer
    network_adam = NeuralNetwork(
        layers=[2, 4, 1],
        activations=['relu', 'sigmoid']
    )
    
    # Train with Adam optimizer
    print("\nTraining with Adam Optimizer:")
    trainer_adam = NetworkTrainer(network_adam, AdamOptimizer(learning_rate=0.1))
    trainer_adam.train(X, y, epochs=1000, verbose=False)
    
    # Make predictions
    predictions_adam, _ = network_adam.forward(X)
    print("Final Predictions:")
    print(predictions_adam)

# Example usage and demonstration
if __name__ == "__main__":
    demonstrate_training()