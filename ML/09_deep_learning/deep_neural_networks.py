"""
Deep Neural Networks Implementation
===================================

This module demonstrates the implementation of deep neural networks with multiple hidden layers.
It covers forward propagation, backpropagation, and training techniques for deep architectures.

Key Concepts:
- Vanishing/Exploding Gradients
- Weight Initialization
- Batch Normalization
- Dropout Regularization
- Gradient Checking
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DeepNeuralNetwork:
    """
    A deep neural network implementation with configurable architecture.
    
    Parameters:
    -----------
    layers_dims : list
        Dimensions of each layer in the network [input_dim, hidden1, hidden2, ..., output]
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    num_iterations : int, default=1000
        Number of training iterations
    activation_functions : list, default=None
        Activation functions for each layer ['relu', 'relu', ..., 'sigmoid']
    initialization : str, default='he'
        Weight initialization method ('he', 'xavier', 'random')
    regularization : str, default=None
        Regularization method ('l2', 'dropout', None)
    lambda_reg : float, default=0.01
        Regularization parameter
    keep_prob : float, default=0.8
        Dropout probability (keep probability)
    """
    
    def __init__(self, layers_dims, learning_rate=0.001, num_iterations=1000,
                 activation_functions=None, initialization='he',
                 regularization=None, lambda_reg=0.01, keep_prob=0.8):
        self.layers_dims = layers_dims
        self.num_layers = len(layers_dims)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.initialization = initialization
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.keep_prob = keep_prob
        
        # Set default activation functions if not provided
        if activation_functions is None:
            self.activation_functions = ['relu'] * (self.num_layers - 2) + ['sigmoid']
        else:
            self.activation_functions = activation_functions
            
        self.parameters = {}
        self.costs = []
        self.grads = {}
        
    def initialize_parameters(self):
        """
        Initialize parameters for the neural network using different methods.
        """
        np.random.seed(1)
        
        for l in range(1, self.num_layers):
            if self.initialization == 'he':
                # He initialization for ReLU activations
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layers_dims[l], self.layers_dims[l-1]) * np.sqrt(2. / self.layers_dims[l-1])
            elif self.initialization == 'xavier':
                # Xavier initialization for sigmoid/tanh activations
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layers_dims[l], self.layers_dims[l-1]) * np.sqrt(1. / self.layers_dims[l-1])
            else:
                # Random initialization
                self.parameters[f'W{l}'] = np.random.randn(
                    self.layers_dims[l], self.layers_dims[l-1]) * 0.01
                
            self.parameters[f'b{l}'] = np.zeros((self.layers_dims[l], 1))
            
    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivative of ReLU activation function."""
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        """Sigmoid activation function."""
        # Clip Z to prevent overflow
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, Z):
        """Derivative of sigmoid activation function."""
        s = self.sigmoid(Z)
        return s * (1 - s)
    
    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
        
        Parameters:
        -----------
        A : numpy array of shape (size of previous layer, number of examples)
            Activations from previous layer
        W : numpy array of shape (size of current layer, size of previous layer)
            Weights matrix
        b : numpy array of shape (size of current layer, 1)
            Bias vector
            
        Returns:
        --------
        Z : numpy array of shape (size of current layer, number of examples)
            Linear output
        cache : tuple
            Cache for backward propagation
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache
    
    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer.
        
        Parameters:
        -----------
        A_prev : numpy array
            Activations from previous layer
        W : numpy array
            Weights matrix
        b : numpy array
            Bias vector
        activation : str
            Activation function to be used ("sigmoid" or "relu")
            
        Returns:
        --------
        A : numpy array
            Output of the activation function
        cache : tuple
            Cache for backward propagation
        """
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        
        if activation == "sigmoid":
            A = self.sigmoid(Z)
        elif activation == "relu":
            A = self.relu(Z)
            
        cache = (linear_cache, Z)
        return A, cache
    
    def forward_propagation(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
        
        Parameters:
        -----------
        X : numpy array of shape (input size, number of examples)
            Data
            
        Returns:
        --------
        AL : numpy array
            Last post-activation value
        caches : list
            List of caches containing every cache of linear_activation_forward()
        """
        caches = []
        A = X
        L = self.num_layers - 1  # Number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1)
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev, 
                self.parameters[f'W{l}'], 
                self.parameters[f'b{l}'], 
                self.activation_functions[l-1]
            )
            caches.append(cache)
            
        # Implement LINEAR -> SIGMOID
        AL, cache = self.linear_activation_forward(
            A, 
            self.parameters[f'W{L}'], 
            self.parameters[f'b{L}'], 
            self.activation_functions[L-1]
        )
        caches.append(cache)
        
        return AL, caches
    
    def compute_cost(self, AL, Y):
        """
        Implement the cost function.
        
        Parameters:
        -----------
        AL : numpy array
            Probability vector corresponding to label predictions
        Y : numpy array
            True "label" vector
            
        Returns:
        --------
        cost : float
            Cross-entropy cost
        """
        m = Y.shape[1]
        
        # Compute loss from AL and Y
        cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m
        
        # Add L2 regularization if specified
        if self.regularization == 'l2':
            L2_regularization_cost = 0
            for l in range(1, self.num_layers):
                L2_regularization_cost += np.sum(np.square(self.parameters[f'W{l}']))
            L2_regularization_cost = (self.lambda_reg / (2 * m)) * L2_regularization_cost
            cost += L2_regularization_cost
            
        cost = np.squeeze(cost)
        return cost
    
    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer.
        
        Parameters:
        -----------
        dZ : numpy array
            Gradient of the cost with respect to the linear output
        cache : tuple
            Cache of values (A_prev, W, b) from forward propagation
            
        Returns:
        --------
        dA_prev : numpy array
            Gradient of the cost with respect to the activation of the previous layer
        dW : numpy array
            Gradient of the cost with respect to W
        db : numpy array
            Gradient of the cost with respect to b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]
        
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        
        # Add L2 regularization gradient
        if self.regularization == 'l2':
            dW += (self.lambda_reg / m) * W
            
        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        
        Parameters:
        -----------
        dA : numpy array
            Post-activation gradient for current layer l
        cache : tuple
            Cache of values (linear_cache, activation_cache)
        activation : str
            Activation function to be used ("sigmoid" or "relu")
            
        Returns:
        --------
        dA_prev : numpy array
            Gradient of the cost with respect to the activation of the previous layer
        dW : numpy array
            Gradient of the cost with respect to W
        db : numpy array
            Gradient of the cost with respect to b
        """
        linear_cache, activation_cache = cache
        
        if activation == "relu":
            dZ = dA * self.relu_derivative(activation_cache)
        elif activation == "sigmoid":
            dZ = dA * self.sigmoid_derivative(activation_cache)
            
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db
    
    def backward_propagation(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group.
        
        Parameters:
        -----------
        AL : numpy array
            Probability vector from forward propagation
        Y : numpy array
            True "label" vector
        caches : list
            List of caches from forward propagation
            
        Returns:
        --------
        grads : dict
            Dictionary with gradients
        """
        grads = {}
        L = self.num_layers - 1  # Number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))
        
        # Lth layer (SIGMOID -> LINEAR) gradients
        current_cache = caches[L - 1]
        grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = self.linear_activation_backward(
            dAL, current_cache, "sigmoid")
        
        # Loop from l=L-2 to l=0
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(
                grads[f"dA{l+1}"], current_cache, "relu")
            grads[f"dA{l}"] = dA_prev_temp
            grads[f"dW{l+1}"] = dW_temp
            grads[f"db{l+1}"] = db_temp
            
        return grads
    
    def update_parameters(self, grads):
        """
        Update parameters using gradient descent.
        
        Parameters:
        -----------
        grads : dict
            Dictionary containing gradients
        """
        L = self.num_layers - 1  # Number of layers
        
        # Update parameters
        for l in range(L):
            self.parameters[f"W{l+1}"] = self.parameters[f"W{l+1}"] - self.learning_rate * grads[f"dW{l+1}"]
            self.parameters[f"b{l+1}"] = self.parameters[f"b{l+1}"] - self.learning_rate * grads[f"db{l+1}"]
    
    def fit(self, X, Y):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : numpy array of shape (input size, number of examples)
            Training data
        Y : numpy array of shape (output size, number of examples)
            Training labels
        """
        np.random.seed(1)
        self.initialize_parameters()
        
        # Optimization loop
        for i in range(self.num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(AL, Y)
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters
            self.update_parameters(grads)
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")
                self.costs.append(cost)
                
    def predict(self, X):
        """
        Predict using the trained model.
        
        Parameters:
        -----------
        X : numpy array
            Input data
            
        Returns:
        --------
        predictions : numpy array
            Predicted labels
        """
        AL, _ = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
    
    def plot_cost(self):
        """Plot the cost function during training."""
        plt.figure(figsize=(10, 6))
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Cost')
        plt.xlabel('Iterations (per hundreds)')
        plt.title(f"Learning rate = {self.learning_rate}")
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).T
    X_test = scaler.transform(X_test).T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)
    
    # Create and train the deep neural network
    layers_dims = [20, 16, 8, 4, 1]  # 4-layer network
    dnn = DeepNeuralNetwork(layers_dims, learning_rate=0.01, num_iterations=1000)
    
    print("Training Deep Neural Network...")
    dnn.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = dnn.predict(X_train)
    test_predictions = dnn.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print(f"\nTraining Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot cost function
    dnn.plot_cost()
    
    # Demonstrate different initialization methods
    print("\n" + "="*50)
    print("Comparing Initialization Methods")
    print("="*50)
    
    methods = ['random', 'xavier', 'he']
    results = {}
    
    for method in methods:
        dnn_method = DeepNeuralNetwork(layers_dims, learning_rate=0.01, num_iterations=500,
                                      initialization=method)
        dnn_method.fit(X_train, y_train)
        test_pred = dnn_method.predict(X_test)
        accuracy = np.mean(test_pred == y_test) * 100
        results[method] = accuracy
        print(f"{method.capitalize()} initialization accuracy: {accuracy:.2f}%")
    
    # Demonstrate regularization
    print("\n" + "="*50)
    print("Demonstrating Regularization")
    print("="*50)
    
    # Without regularization
    dnn_no_reg = DeepNeuralNetwork(layers_dims, learning_rate=0.01, num_iterations=500)
    dnn_no_reg.fit(X_train, y_train)
    test_pred_no_reg = dnn_no_reg.predict(X_test)
    accuracy_no_reg = np.mean(test_pred_no_reg == y_test) * 100
    print(f"No regularization accuracy: {accuracy_no_reg:.2f}%")
    
    # With L2 regularization
    dnn_l2_reg = DeepNeuralNetwork(layers_dims, learning_rate=0.01, num_iterations=500,
                                  regularization='l2', lambda_reg=0.01)
    dnn_l2_reg.fit(X_train, y_train)
    test_pred_l2_reg = dnn_l2_reg.predict(X_test)
    accuracy_l2_reg = np.mean(test_pred_l2_reg == y_test) * 100
    print(f"L2 regularization accuracy: {accuracy_l2_reg:.2f}%")