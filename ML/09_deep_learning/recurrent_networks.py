"""
Recurrent Neural Networks Implementation
=======================================

This module demonstrates the implementation of Recurrent Neural Networks (RNNs) and LSTM networks
for sequential data processing tasks. It covers basic RNN cells, LSTM cells, and their applications.

Key Concepts:
- Recurrent Neural Networks
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)
- Sequence-to-Sequence Models
- Time Series Forecasting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler


class RNNCell:
    """
    A basic RNN cell implementation.
    
    Parameters:
    -----------
    input_size : int
        Size of input features
    hidden_size : int
        Size of hidden state
    output_size : int
        Size of output
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.1  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
        
    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation function."""
        return 1 - np.tanh(x) ** 2
    
    def forward(self, x, h_prev):
        """
        Forward pass of RNN cell.
        
        Parameters:
        -----------
        x : numpy array of shape (input_size, 1)
            Input at current time step
        h_prev : numpy array of shape (hidden_size, 1)
            Hidden state from previous time step
            
        Returns:
        --------
        h : numpy array of shape (hidden_size, 1)
            Hidden state at current time step
        y : numpy array of shape (output_size, 1)
            Output at current time step
        """
        # Compute hidden state
        h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        
        # Compute output
        y = np.dot(self.Why, h) + self.by
        
        return h, y
    
    def predict_sequence(self, x_sequence):
        """
        Predict output sequence for input sequence.
        
        Parameters:
        -----------
        x_sequence : numpy array of shape (sequence_length, input_size)
            Input sequence
            
        Returns:
        --------
        y_sequence : numpy array of shape (sequence_length, output_size)
            Output sequence
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        y_sequence = []
        
        for x in x_sequence:
            x = x.reshape(-1, 1)  # Reshape to column vector
            h, y = self.forward(x, h)
            y_sequence.append(y.flatten())
            
        return np.array(y_sequence)


class LSTMCell:
    """
    An LSTM cell implementation.
    
    Parameters:
    -----------
    input_size : int
        Size of input features
    hidden_size : int
        Size of hidden state
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for input, forget, gate, and output gates
        # Input gate
        self.Wxi = np.random.randn(hidden_size, input_size) * 0.1
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.Wxf = np.random.randn(hidden_size, input_size) * 0.1
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        
        # Gate (candidate values)
        self.Wxg = np.random.randn(hidden_size, input_size) * 0.1
        self.Whg = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bg = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wxo = np.random.randn(hidden_size, input_size) * 0.1
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))
        
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass of LSTM cell.
        
        Parameters:
        -----------
        x : numpy array of shape (input_size, 1)
            Input at current time step
        h_prev : numpy array of shape (hidden_size, 1)
            Hidden state from previous time step
        c_prev : numpy array of shape (hidden_size, 1)
            Cell state from previous time step
            
        Returns:
        --------
        h : numpy array of shape (hidden_size, 1)
            Hidden state at current time step
        c : numpy array of shape (hidden_size, 1)
            Cell state at current time step
        """
        x = x.reshape(-1, 1)  # Ensure x is a column vector
        
        # Input gate
        i = self.sigmoid(np.dot(self.Wxi, x) + np.dot(self.Whi, h_prev) + self.bi)
        
        # Forget gate
        f = self.sigmoid(np.dot(self.Wxf, x) + np.dot(self.Whf, h_prev) + self.bf)
        
        # Gate (candidate values)
        g = self.tanh(np.dot(self.Wxg, x) + np.dot(self.Whg, h_prev) + self.bg)
        
        # Cell state
        c = f * c_prev + i * g
        
        # Output gate
        o = self.sigmoid(np.dot(self.Wxo, x) + np.dot(self.Who, h_prev) + self.bo)
        
        # Hidden state
        h = o * self.tanh(c)
        
        return h, c


class SequenceToSequence:
    """
    A simple sequence-to-sequence model using RNN encoder-decoder architecture.
    
    Parameters:
    -----------
    input_size : int
        Size of input features
    hidden_size : int
        Size of hidden state
    output_size : int
        Size of output features
    sequence_length : int
        Length of input/output sequences
    """
    
    def __init__(self, input_size, hidden_size, output_size, sequence_length):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        
        # Encoder RNN
        self.encoder = RNNCell(input_size, hidden_size, hidden_size)
        
        # Decoder RNN
        self.decoder = RNNCell(output_size, hidden_size, output_size)
        
    def train(self, X, Y, learning_rate=0.01, epochs=100):
        """
        Train the sequence-to-sequence model.
        
        Parameters:
        -----------
        X : numpy array of shape (num_samples, sequence_length, input_size)
            Input sequences
        Y : numpy array of shape (num_samples, sequence_length, output_size)
            Target sequences
        learning_rate : float
            Learning rate for training
        epochs : int
            Number of training epochs
        """
        print("Training sequence-to-sequence model...")
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                # Encoder forward pass
                h_encoder = np.zeros((self.hidden_size, 1))
                for t in range(self.sequence_length):
                    x_t = X[i, t].reshape(-1, 1)
                    h_encoder, _ = self.encoder.forward(x_t, h_encoder)
                
                # Decoder forward pass
                h_decoder = h_encoder  # Initial hidden state from encoder
                c_decoder = np.zeros((self.hidden_size, 1))  # Initial cell state
                decoder_outputs = []
                
                # Use teacher forcing during training
                for t in range(self.sequence_length):
                    if t == 0:
                        # First decoder input is usually zeros or start token
                        y_prev = np.zeros((self.output_size, 1))
                    else:
                        y_prev = Y[i, t-1].reshape(-1, 1)
                    
                    h_decoder, y_t = self.decoder.forward(y_prev, h_decoder)
                    decoder_outputs.append(y_t.flatten())
                
                # Compute loss
                decoder_outputs = np.array(decoder_outputs)
                loss = np.mean((decoder_outputs - Y[i]) ** 2)
                total_loss += loss
                
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
                
        return losses


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample time series data
    np.random.seed(42)
    
    # Create a synthetic time series
    t = np.linspace(0, 10, 100)
    ts_data = np.sin(t) + 0.1 * np.random.randn(100)
    
    # Prepare data for RNN (sequence prediction)
    sequence_length = 10
    X, Y = [], []
    
    for i in range(len(ts_data) - sequence_length):
        X.append(ts_data[i:i+sequence_length])
        Y.append(ts_data[i+sequence_length])
    
    X = np.array(X)
    Y = np.array(Y)
    
    # Reshape for RNN input (samples, sequence_length, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = Y.reshape(Y.shape[0], 1)
    
    print("Time Series Data Preparation:")
    print(f"Input sequences shape: {X.shape}")
    print(f"Target values shape: {Y.shape}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    # Demonstrate basic RNN cell
    print("\n" + "="*50)
    print("Basic RNN Cell Demonstration")
    print("="*50)
    
    rnn_cell = RNNCell(input_size=1, hidden_size=16, output_size=1)
    
    # Predict a sequence
    sample_sequence = X_test[0]  # Take first test sequence
    predictions = rnn_cell.predict_sequence(sample_sequence)
    
    print(f"Sample input sequence shape: {sample_sequence.shape}")
    print(f"Sample output sequence shape: {predictions.shape}")
    print(f"First few predictions: {predictions.flatten()[:5]}")
    
    # Demonstrate LSTM cell
    print("\n" + "="*50)
    print("LSTM Cell Demonstration")
    print("="*50)
    
    lstm_cell = LSTMCell(input_size=1, hidden_size=16)
    
    # Process a sequence with LSTM
    h = np.zeros((16, 1))  # Initial hidden state
    c = np.zeros((16, 1))  # Initial cell state
    
    print("Processing sequence with LSTM:")
    for i, x_val in enumerate(sample_sequence[:5]):  # Process first 5 elements
        x = np.array([[x_val]])  # Reshape to (1, 1)
        h, c = lstm_cell.forward(x, h, c)
        print(f"Time step {i+1}: Input={x_val:.4f}, Hidden state mean={np.mean(h):.4f}")
    
    # Demonstrate sequence-to-sequence model
    print("\n" + "="*50)
    print("Sequence-to-Sequence Model Demonstration")
    print("="*50)
    
    # Prepare data for sequence-to-sequence (copy task)
    seq_len = 5
    num_samples = 100
    
    # Generate random sequences
    X_seq2seq = np.random.randn(num_samples, seq_len, 1)
    Y_seq2seq = X_seq2seq.copy()  # Copy task - output same as input
    
    print(f"Sequence-to-sequence data shape: {X_seq2seq.shape}")
    
    # Create and train model
    seq2seq = SequenceToSequence(input_size=1, hidden_size=32, output_size=1, sequence_length=seq_len)
    
    # Note: Training is simplified for demonstration
    print("Model created successfully")
    print(f"Encoder hidden size: {seq2seq.encoder.hidden_size}")
    print(f"Decoder hidden size: {seq2seq.decoder.hidden_size}")
    
    # Demonstrate vanishing gradient problem
    print("\n" + "="*50)
    print("Vanishing Gradient Problem Demonstration")
    print("="*50)
    
    # Create a long sequence to show gradient issues
    long_sequence = np.array([[np.sin(i/10) for i in range(50)]])
    long_sequence = long_sequence.reshape(long_sequence.shape[1], 1)
    
    print("Processing long sequence with basic RNN:")
    print(f"Sequence length: {len(long_sequence)}")
    
    h = np.zeros((16, 1))
    hidden_states = []
    
    rnn_long = RNNCell(input_size=1, hidden_size=16, output_size=1)
    
    for i, x_val in enumerate(long_sequence[:10]):  # Process first 10 elements
        x = np.array([[x_val[0]]])
        h, _ = rnn_long.forward(x, h)
        hidden_states.append(np.mean(np.abs(h)))
        print(f"Time step {i+1}: Mean hidden state magnitude = {hidden_states[-1]:.6f}")
    
    # Compare with LSTM
    print("\nProcessing same sequence with LSTM:")
    h = np.zeros((16, 1))
    c = np.zeros((16, 1))
    lstm_hidden_states = []
    
    lstm_long = LSTMCell(input_size=1, hidden_size=16)
    
    for i, x_val in enumerate(long_sequence[:10]):
        x = np.array([[x_val[0]]])
        h, c = lstm_long.forward(x, h, c)
        lstm_hidden_states.append(np.mean(np.abs(h)))
        print(f"Time step {i+1}: Mean hidden state magnitude = {lstm_hidden_states[-1]:.6f}")
    
    print("\n" + "="*50)
    print("Key Differences Summary:")
    print("="*50)
    print("1. Basic RNN: Simple but suffers from vanishing gradient problem")
    print("2. LSTM: Uses gates to control information flow, better for long sequences")
    print("3. GRU: Simplified version of LSTM with fewer parameters")
    print("4. Applications: Time series forecasting, NLP, speech recognition")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow/Keras: tf.keras.layers.LSTM, tf.keras.layers.GRU")
    print("- PyTorch: torch.nn.LSTM, torch.nn.GRU")
    print("- These provide GPU acceleration and optimized implementations")