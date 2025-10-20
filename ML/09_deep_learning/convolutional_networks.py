"""
Convolutional Neural Networks Implementation
==========================================

This module demonstrates the implementation of Convolutional Neural Networks (CNNs) for image processing tasks.
It covers convolution operations, pooling, and building complete CNN architectures.

Key Concepts:
- Convolutional Layers
- Pooling Layers
- Padding and Stride
- Feature Maps
- CNN Architectures
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    
    Parameters:
    -----------
    X : numpy array of shape (m, n_H, n_W, n_C)
        Array of m images
    pad : int
        Amount of padding around each image on vertical and horizontal dimensions
        
    Returns:
    --------
    X_pad : numpy array
        Padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice of the output activation.
    
    Parameters:
    -----------
    a_slice_prev : numpy array of shape (f, f, n_C_prev)
        Slice of input data
    W : numpy array of shape (f, f, n_C_prev)
        Weight parameters contained in a window
    b : numpy array of shape (1, 1, 1)
        Bias parameter
        
    Returns:
    --------
    Z : float
        Result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice_prev and W
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s
    Z = np.sum(s)
    # Add bias b to Z
    Z = Z + float(b)
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function.
    
    Parameters:
    -----------
    A_prev : numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Output activations of the previous layer
    W : numpy array of shape (f, f, n_C_prev, n_C)
        Weights
    b : numpy array of shape (1, 1, 1, n_C)
        Biases
    hparameters : dict
        "stride" and "pad"
        
    Returns:
    --------
    Z : numpy array of shape (m, n_H, n_W, n_C)
        Convolution output
    cache : tuple
        Cache of values needed for the conv_backward() function
    """
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # Initialize the output volume Z with zeros
    Z = np.zeros((m, n_H, n_W, n_C))
    
    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):  # Loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
        for h in range(n_H):  # Loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):  # Loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):  # Loop over channels (= #filters) of the output volume
                    # Use the corners to define the (3D) slice of a_prev_pad
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
    
    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer.
    
    Parameters:
    -----------
    A_prev : numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Input data
    hparameters : dict
        "f" and "stride"
    mode : str
        "max" or "average"
        
    Returns:
    --------
    A : numpy array of shape (m, n_H, n_W, n_C)
        Output of the pool layer
    cache : tuple
        Cache used in the backward pass
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):  # Loop over the training examples
        for h in range(n_H):  # Loop on the vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):  # Loop on the horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):  # Loop over the channels of the output volume
                    # Use the corners to define the current slice on the ith training example of A_prev
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    
                    # Compute the pooling operation on the slice
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    return A, cache


class SimpleCNN:
    """
    A simple CNN implementation for educational purposes.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    conv_filters : int
        Number of filters in the convolutional layer
    conv_filter_size : int
        Size of convolutional filters
    pool_size : int
        Size of pooling window
    hidden_units : int
        Number of units in the hidden fully connected layer
    num_classes : int
        Number of output classes
    learning_rate : float
        Learning rate for training
    """
    
    def __init__(self, input_shape=(28, 28, 1), conv_filters=8, conv_filter_size=5,
                 pool_size=2, hidden_units=128, num_classes=10, learning_rate=0.001):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_filter_size = conv_filter_size
        self.pool_size = pool_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize CNN parameters."""
        # Convolutional layer parameters
        self.W_conv = np.random.randn(self.conv_filter_size, self.conv_filter_size,
                                     self.input_shape[2], self.conv_filters) * 0.1
        self.b_conv = np.zeros((1, 1, 1, self.conv_filters))
        
        # Calculate dimensions after conv and pooling
        conv_output_size = (self.input_shape[0] - self.conv_filter_size + 1) // self.pool_size
        flattened_size = conv_output_size * conv_output_size * self.conv_filters
        
        # Fully connected layer parameters
        self.W_fc1 = np.random.randn(flattened_size, self.hidden_units) * 0.1
        self.b_fc1 = np.zeros((1, self.hidden_units))
        
        # Output layer parameters
        self.W_fc2 = np.random.randn(self.hidden_units, self.num_classes) * 0.1
        self.b_fc2 = np.zeros((1, self.num_classes))
        
    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivative of ReLU activation function."""
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        """Softmax activation function."""
        # Subtract max for numerical stability
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Implement forward propagation for the CNN.
        
        Parameters:
        -----------
        X : numpy array of shape (m, height, width, channels)
            Input data
            
        Returns:
        --------
        caches : dict
            Dictionary containing all caches for backward propagation
        """
        caches = {}
        
        # Convolutional layer
        hparameters_conv = {"pad": 0, "stride": 1}
        Z_conv, cache_conv = conv_forward(X, self.W_conv, self.b_conv, hparameters_conv)
        A_conv = self.relu(Z_conv)
        caches['conv'] = cache_conv
        
        # Pooling layer
        hparameters_pool = {"f": self.pool_size, "stride": self.pool_size}
        A_pool, cache_pool = pool_forward(A_conv, hparameters_pool, mode="max")
        caches['pool'] = cache_pool
        
        # Flatten for fully connected layers
        m = X.shape[0]
        A_flat = A_pool.reshape(m, -1)
        caches['flat_shape'] = A_pool.shape
        
        # First fully connected layer
        Z_fc1 = np.dot(A_flat, self.W_fc1) + self.b_fc1
        A_fc1 = self.relu(Z_fc1)
        caches['fc1'] = (A_flat, Z_fc1)
        
        # Output layer
        Z_fc2 = np.dot(A_fc1, self.W_fc2) + self.b_fc2
        A_fc2 = self.softmax(Z_fc2)
        caches['fc2'] = (A_fc1, Z_fc2)
        
        return A_fc2, caches
    
    def compute_cost(self, AL, Y):
        """
        Compute the cross-entropy cost.
        
        Parameters:
        -----------
        AL : numpy array
            Probability vector corresponding to label predictions
        Y : numpy array
            True "label" vector (one-hot encoded)
            
        Returns:
        --------
        cost : float
            Cross-entropy cost
        """
        m = Y.shape[0]
        # Compute loss from AL and Y
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m
        cost = np.squeeze(cost)
        return cost
    
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
        predictions = np.argmax(AL, axis=1)
        return predictions


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data (simulating image data)
    np.random.seed(42)
    m = 100  # Number of examples
    height, width, channels = 28, 28, 1
    num_classes = 10
    
    # Create sample data
    X = np.random.randn(m, height, width, channels)
    y = np.random.randint(0, num_classes, (m,))
    
    # One-hot encode labels
    Y = np.eye(num_classes)[y]
    
    # Split data
    X_train, X_test = X[:80], X[80:]
    Y_train, Y_test = Y[:80], Y[80:]
    y_train, y_test = y[:80], y[80:]
    
    print("Training Simple CNN...")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: {num_classes}")
    
    # Create and train CNN
    cnn = SimpleCNN(input_shape=(height, width, channels), num_classes=num_classes)
    
    # Forward propagation example
    print("\nPerforming forward propagation...")
    AL, caches = cnn.forward_propagation(X_train[:5])  # Use first 5 examples
    print(f"Output shape: {AL.shape}")
    print(f"Sample predictions probabilities:\n{AL[0]}")
    
    # Compute cost
    cost = cnn.compute_cost(AL, Y_train[:5])
    print(f"Sample cost: {cost:.4f}")
    
    # Make predictions
    predictions = cnn.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100
    print(f"\nSample test accuracy: {accuracy:.2f}%")
    
    # Demonstrate convolution operation
    print("\n" + "="*50)
    print("Convolution Operation Demonstration")
    print("="*50)
    
    # Create a simple image and filter
    image = np.random.randn(1, 10, 10, 1)
    filter_weights = np.random.randn(3, 3, 1, 1)
    filter_bias = np.zeros((1, 1, 1, 1))
    
    hparameters = {"pad": 0, "stride": 1}
    Z, _ = conv_forward(image, filter_weights, filter_bias, hparameters)
    
    print(f"Input image shape: {image.shape}")
    print(f"Filter shape: {filter_weights.shape}")
    print(f"Output feature map shape: {Z.shape}")
    
    # Demonstrate pooling operation
    print("\n" + "="*50)
    print("Pooling Operation Demonstration")
    print("="*50)
    
    hparameters_pool = {"f": 2, "stride": 2}
    A_pool, _ = pool_forward(Z, hparameters_pool, mode="max")
    
    print(f"Input to pooling shape: {Z.shape}")
    print(f"Output from pooling shape: {A_pool.shape}")
    
    # Visualize feature maps
    print("\n" + "="*50)
    print("Feature Map Visualization")
    print("="*50)
    
    # Create a sample image with a pattern
    sample_image = np.zeros((1, 8, 8, 1))
    sample_image[0, 2:6, 2:6, 0] = 1  # White square in the middle
    
    # Simple edge detection filter
    edge_filter = np.array([[[[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]]]]).transpose(1, 2, 3, 0)
    
    edge_bias = np.zeros((1, 1, 1, 1))
    hparams = {"pad": 1, "stride": 1}
    
    edge_response, _ = conv_forward(sample_image, edge_filter, edge_bias, hparams)
    
    print("Sample image and edge detection filter applied:")
    print("Original image shape:", sample_image.shape)
    print("Edge detection filter shape:", edge_filter.shape)
    print("Edge response shape:", edge_response.shape)
    
    # Note: In practice, you would use libraries like TensorFlow or PyTorch
    # for efficient CNN implementation with GPU support
    print("\n" + "="*50)
    print("Note: This is a simplified implementation for educational purposes.")
    print("For production use, consider using TensorFlow, PyTorch, or Keras.")
    print("="*50)