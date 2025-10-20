"""
Autoencoders for Dimensionality Reduction Implementation
===============================================

This module provides implementations of autoencoders for nonlinear 
dimensionality reduction using neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Autoencoder functionality will be limited.")


class Autoencoder:
    """
    Autoencoder for dimensionality reduction.
    
    Parameters:
    -----------
    n_components : int, default=2
        Dimension of the encoded space.
    hidden_layers : list, default=[64, 32]
        Number of units in hidden layers.
    activation : str, default='relu'
        Activation function for hidden layers.
    optimizer : str, default='adam'
        Optimizer for training.
    loss : str, default='mse'
        Loss function for training.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes:
    -----------
    encoder : keras.Model
        Encoder part of the autoencoder.
    decoder : keras.Model
        Decoder part of the autoencoder.
    autoencoder : keras.Model
        Full autoencoder model.
    history : dict
        Training history.
    """
    
    def __init__(self, n_components=2, hidden_layers=[64, 32], activation='relu',
                 optimizer='adam', loss='mse', epochs=100, batch_size=32, 
                 random_state=None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder functionality.")
        
        self.n_components = n_components
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        if random_state:
            tf.random.set_seed(random_state)
    
    def _build_encoder(self, input_dim):
        """Build encoder network."""
        inputs = keras.Input(shape=(input_dim,))
        
        # Hidden layers
        x = inputs
        for units in self.hidden_layers:
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Encoding layer
        encoded = layers.Dense(self.n_components, activation='linear')(x)
        
        encoder = keras.Model(inputs, encoded, name="encoder")
        return encoder
    
    def _build_decoder(self, output_dim):
        """Build decoder network."""
        inputs = keras.Input(shape=(self.n_components,))
        
        # Hidden layers (reverse order)
        x = inputs
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation=self.activation)(x)
        
        # Output layer
        decoded = layers.Dense(output_dim, activation='linear')(x)
        
        decoder = keras.Model(inputs, decoded, name="decoder")
        return decoder
    
    def fit(self, X, validation_split=0.1):
        """
        Train the autoencoder.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        validation_split : float, default=0.1
            Fraction of data to use for validation.
            
        Returns:
        --------
        self : Autoencoder
            Fitted autoencoder.
        """
        # Normalize data to [0, 1] range
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        n_features = X_scaled.shape[1]
        
        # Build encoder and decoder
        self.encoder = self._build_encoder(n_features)
        self.decoder = self._build_decoder(n_features)
        
        # Build full autoencoder
        inputs = keras.Input(shape=(n_features,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.autoencoder = keras.Model(inputs, decoded, name="autoencoder")
        
        # Compile model
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)
        
        # Train model
        self.history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        return self
    
    def transform(self, X):
        """
        Encode data to lower-dimensional space.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to encode.
            
        Returns:
        --------
        X_encoded : array, shape (n_samples, n_components)
            Encoded data.
        """
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled)
    
    def inverse_transform(self, X_encoded):
        """
        Decode data from lower-dimensional space.
        
        Parameters:
        -----------
        X_encoded : array-like, shape (n_samples, n_components)
            Encoded data.
            
        Returns:
        --------
        X_decoded : array, shape (n_samples, n_features)
            Decoded data.
        """
        X_decoded = self.decoder.predict(X_encoded)
        return self.scaler.inverse_transform(X_decoded)
    
    def fit_transform(self, X, validation_split=0.1):
        """
        Train the autoencoder and encode the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        validation_split : float, default=0.1
            Fraction of data to use for validation.
            
        Returns:
        --------
        X_encoded : array, shape (n_samples, n_components)
            Encoded data.
        """
        return self.fit(X, validation_split).transform(X)
    
    def plot_loss(self):
        """Plot training and validation loss."""
        if not hasattr(self, 'history'):
            print("Model not trained yet.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def compare_autoencoder_with_pca(X, y, n_components=2):
    """
    Compare autoencoder with PCA for dimensionality reduction.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to reduce.
    y : array-like, shape (n_samples,)
        Class labels.
    n_components : int, default=2
        Number of components.
        
    Returns:
    --------
    results : dict
        Dictionary containing results from both methods.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    results = {
        'pca': {
            'embedding': X_pca,
            'explained_variance_ratio': pca.explained_variance_ratio_
        }
    }
    
    # Apply autoencoder (if TensorFlow available)
    if TENSORFLOW_AVAILABLE:
        autoencoder = Autoencoder(n_components=n_components, epochs=50, random_state=42)
        X_autoencoder = autoencoder.fit_transform(X_scaled)
        
        results['autoencoder'] = {
            'embedding': X_autoencoder,
            'model': autoencoder
        }
    
    return results


def visualize_autoencoder_results(X, y, title="Autoencoder Visualization"):
    """
    Visualize autoencoder results with class labels.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to encode.
    y : array-like, shape (n_samples,)
        Class labels.
    title : str, default="Autoencoder Visualization"
        Title for the plot.
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available for autoencoder visualization.")
        return
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply autoencoder
    autoencoder = Autoencoder(n_components=2, epochs=50, random_state=42)
    X_encoded = autoencoder.fit_transform(X_scaled)
    
    # Plot results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Autoencoder Dimension 1')
    plt.ylabel('Autoencoder Dimension 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot training history
    autoencoder.plot_loss()
    
    return autoencoder


def demonstrate_reconstruction(X, n_samples=5):
    """
    Demonstrate autoencoder reconstruction capability.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to reconstruct.
    n_samples : int, default=5
        Number of samples to show.
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available for reconstruction demonstration.")
        return
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply autoencoder
    autoencoder = Autoencoder(n_components=2, epochs=50, random_state=42)
    autoencoder.fit(X_scaled)
    
    # Reconstruct samples
    X_reconstructed = autoencoder.inverse_transform(autoencoder.transform(X_scaled[:n_samples]))
    
    # Plot original vs reconstructed
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(X_scaled[i].reshape(8, 8), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(X_reconstructed[i].reshape(8, 8), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Load sample data - digits dataset for better visualization
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print("Autoencoder for Dimensionality Reduction:")
    print(f"Original dimensions: {X.shape}")
    
    # Compare autoencoder with PCA
    print("\nComparing autoencoder with PCA...")
    results = compare_autoencoder_with_pca(X, y, n_components=2)
    
    print("PCA results:")
    print(f"  Reduced dimensions: {results['pca']['embedding'].shape}")
    print(f"  Explained variance ratio: {results['pca']['explained_variance_ratio']}")
    
    if 'autoencoder' in results:
        print("Autoencoder results:")
        print(f"  Reduced dimensions: {results['autoencoder']['embedding'].shape}")
        print("  Model trained successfully")
    
    # Visualize autoencoder results (if TensorFlow available)
    print("\nGenerating autoencoder visualization...")
    if TENSORFLOW_AVAILABLE:
        autoencoder_model = visualize_autoencoder_results(X, y, "Digits Dataset Autoencoder Projection")
    
    # Demonstrate reconstruction
    print("\nDemonstrating reconstruction capability...")
    if TENSORFLOW_AVAILABLE:
        demonstrate_reconstruction(X)
    
    # Demonstrate with iris dataset
    print("\nIris Dataset Example:")
    X_iris, y_iris = load_iris(return_X_y=True)
    scaler_iris = StandardScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris)
    
    if TENSORFLOW_AVAILABLE:
        visualize_autoencoder_results(X_iris_scaled, y_iris, "Iris Dataset Autoencoder Projection")
    
    print("\nKey Points about Autoencoders:")
    print("• Autoencoders can learn nonlinear mappings for dimensionality reduction")
    print("• They can capture complex patterns that linear methods like PCA miss")
    print("• Require more computational resources and training time than PCA")
    print("• Need careful tuning of architecture and hyperparameters")
    print("• Require TensorFlow/Keras: pip install tensorflow")