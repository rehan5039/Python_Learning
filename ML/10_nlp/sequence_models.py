"""
Sequence Models for NLP
======================

This module demonstrates sequence models for NLP tasks including RNNs, LSTMs, and GRUs.
It covers implementation of recurrent architectures for text processing.

Key Concepts:
- Recurrent Neural Networks for Text
- LSTM and GRU Architectures
- Sequence-to-Sequence Models
- Text Generation
- Named Entity Recognition
"""

import numpy as np
from collections import defaultdict, Counter
import random


class SimpleRNN:
    """
    A simple Recurrent Neural Network for sequence modeling.
    
    Parameters:
    -----------
    input_size : int
        Size of input vocabulary
    hidden_size : int
        Size of hidden state
    output_size : int
        Size of output vocabulary
    learning_rate : float, default=0.01
        Learning rate for training
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
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
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward(self, inputs):
        """
        Forward pass through the RNN.
        
        Parameters:
        -----------
        inputs : list
            List of input indices
            
        Returns:
        --------
        outputs : list
            List of output probabilities
        hidden_states : list
            List of hidden states
        """
        # Initialize hidden state
        h = np.zeros((self.hidden_size, 1))
        
        # Store outputs and hidden states
        outputs = []
        hidden_states = [h]
        
        # Forward pass through sequence
        for t in range(len(inputs)):
            # One-hot encode input
            x = np.zeros((self.input_size, 1))
            x[inputs[t]] = 1
            
            # Compute hidden state
            h = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            hidden_states.append(h)
            
            # Compute output
            y = np.dot(self.Why, h) + self.by
            p = self.softmax(y)
            outputs.append(p)
        
        return outputs, hidden_states
    
    def predict(self, inputs):
        """
        Predict output sequence.
        
        Parameters:
        -----------
        inputs : list
            List of input indices
            
        Returns:
        --------
        predictions : list
            List of predicted output indices
        """
        outputs, _ = self.forward(inputs)
        predictions = [np.argmax(output) for output in outputs]
        return predictions


class SimpleLSTM:
    """
    A simple Long Short-Term Memory network for sequence modeling.
    
    Parameters:
    -----------
    input_size : int
        Size of input vocabulary
    hidden_size : int
        Size of hidden state
    output_size : int
        Size of output vocabulary
    learning_rate : float, default=0.01
        Learning rate for training
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize LSTM weights
        # Input gate
        self.Wxi = np.random.randn(hidden_size, input_size) * 0.1
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bi = np.zeros((hidden_size, 1))
        
        # Forget gate
        self.Wxf = np.random.randn(hidden_size, input_size) * 0.1
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bf = np.zeros((hidden_size, 1))
        
        # Output gate
        self.Wxo = np.random.randn(hidden_size, input_size) * 0.1
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bo = np.zeros((hidden_size, 1))
        
        # Cell candidate
        self.Wxg = np.random.randn(hidden_size, input_size) * 0.1
        self.Whg = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bg = np.zeros((hidden_size, 1))
        
        # Output layer
        self.Why = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def forward(self, inputs):
        """
        Forward pass through the LSTM.
        
        Parameters:
        -----------
        inputs : list
            List of input indices
            
        Returns:
        --------
        outputs : list
            List of output probabilities
        hidden_states : list
            List of hidden states
        cell_states : list
            List of cell states
        """
        # Initialize hidden and cell states
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        # Store outputs and states
        outputs = []
        hidden_states = [h]
        cell_states = [c]
        
        # Forward pass through sequence
        for t in range(len(inputs)):
            # One-hot encode input
            x = np.zeros((self.input_size, 1))
            x[inputs[t]] = 1
            
            # LSTM computations
            # Input gate
            i = self.sigmoid(np.dot(self.Wxi, x) + np.dot(self.Whi, h) + self.bi)
            
            # Forget gate
            f = self.sigmoid(np.dot(self.Wxf, x) + np.dot(self.Whf, h) + self.bf)
            
            # Output gate
            o = self.sigmoid(np.dot(self.Wxo, x) + np.dot(self.Who, h) + self.bo)
            
            # Cell candidate
            g = self.tanh(np.dot(self.Wxg, x) + np.dot(self.Whg, h) + self.bg)
            
            # Cell state
            c = f * c + i * g
            
            # Hidden state
            h = o * self.tanh(c)
            
            # Store states
            hidden_states.append(h)
            cell_states.append(c)
            
            # Compute output
            y = np.dot(self.Why, h) + self.by
            p = self.softmax(y)
            outputs.append(p)
        
        return outputs, hidden_states, cell_states
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        x = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def predict(self, inputs):
        """
        Predict output sequence.
        
        Parameters:
        -----------
        inputs : list
            List of input indices
            
        Returns:
        --------
        predictions : list
            List of predicted output indices
        """
        outputs, _, _ = self.forward(inputs)
        predictions = [np.argmax(output) for output in outputs]
        return predictions


class SequenceToSequence:
    """
    Sequence-to-sequence model with encoder-decoder architecture.
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    hidden_size : int
        Size of hidden state
    learning_rate : float, default=0.01
        Learning rate for training
    """
    
    def __init__(self, vocab_size, hidden_size, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Encoder (LSTM)
        self.encoder = SimpleLSTM(vocab_size, hidden_size, hidden_size, learning_rate)
        
        # Decoder (LSTM)
        self.decoder = SimpleLSTM(vocab_size, hidden_size, vocab_size, learning_rate)
    
    def encode(self, input_sequence):
        """
        Encode input sequence.
        
        Parameters:
        -----------
        input_sequence : list
            Input sequence of indices
            
        Returns:
        --------
        hidden_state : numpy array
            Final hidden state
        cell_state : numpy array
            Final cell state
        """
        _, hidden_states, cell_states = self.encoder.forward(input_sequence)
        return hidden_states[-1], cell_states[-1]
    
    def decode(self, encoder_hidden, encoder_cell, target_sequence=None, max_length=20):
        """
        Decode sequence from encoder states.
        
        Parameters:
        -----------
        encoder_hidden : numpy array
            Encoder final hidden state
        encoder_cell : numpy array
            Encoder final cell state
        target_sequence : list, optional
            Target sequence for training (teacher forcing)
        max_length : int, default=20
            Maximum output length
            
        Returns:
        --------
        outputs : list
            Output probabilities
        """
        # Initialize decoder with encoder states
        h = encoder_hidden
        c = encoder_cell
        
        outputs = []
        input_idx = 0  # Start token index (simplified)
        
        # Generate sequence
        for t in range(max_length):
            # One-hot encode input
            x = np.zeros((self.vocab_size, 1))
            x[input_idx] = 1
            
            # LSTM computations (simplified decoder step)
            # In practice, this would be a full LSTM forward pass
            # For demonstration, we'll use a simplified approach
            
            # Compute output
            y = np.dot(self.decoder.Why, h) + self.decoder.by
            p = self.decoder.softmax(y)
            outputs.append(p)
            
            # Update input for next step (greedy decoding)
            input_idx = np.argmax(p)
            
            # Break if end token (simplified)
            if input_idx == 1:  # End token index
                break
        
        return outputs


class TextGenerator:
    """
    Text generation using trained sequence models.
    
    Parameters:
    -----------
    model : object
        Trained sequence model
    idx_to_char : dict
        Mapping from indices to characters
    char_to_idx : dict
        Mapping from characters to indices
    """
    
    def __init__(self, model, idx_to_char, char_to_idx):
        self.model = model
        self.idx_to_char = idx_to_char
        self.char_to_idx = char_to_idx
    
    def generate_text(self, seed_text, length=100, temperature=1.0):
        """
        Generate text using the trained model.
        
        Parameters:
        -----------
        seed_text : str
            Initial text to start generation
        length : int, default=100
            Length of generated text
        temperature : float, default=1.0
            Sampling temperature (lower = more conservative)
            
        Returns:
        --------
        generated_text : str
            Generated text
        """
        # Convert seed text to indices
        indices = [self.char_to_idx.get(char, 0) for char in seed_text]
        
        # Generate text
        for _ in range(length):
            # Get model predictions
            if hasattr(self.model, 'forward'):
                outputs, _ = self.model.forward(indices[-50:])  # Use last 50 chars
                probs = outputs[-1].flatten()
            else:
                # Simplified prediction for demonstration
                probs = np.ones(len(self.idx_to_char)) / len(self.idx_to_char)
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.log(probs + 1e-8) / temperature
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
            
            # Sample next character
            next_idx = np.random.choice(len(probs), p=probs)
            indices.append(next_idx)
        
        # Convert indices back to text
        generated_text = ''.join([self.idx_to_char.get(idx, '') for idx in indices])
        return generated_text


# Example usage and demonstration
if __name__ == "__main__":
    # Sample data for demonstration
    print("Sequence Models for NLP Demonstration")
    print("=" * 50)
    
    # Sample text for character-level modeling
    sample_text = "hello world this is a sample text for sequence modeling"
    
    # Create character mappings
    chars = list(set(sample_text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars}")
    
    # Simple RNN demonstration
    print("\n1. Simple RNN:")
    rnn = SimpleRNN(input_size=vocab_size, hidden_size=20, output_size=vocab_size)
    
    # Convert text to indices
    text_indices = [char_to_idx[ch] for ch in "hello"]
    print(f"Input sequence: {text_indices}")
    
    # Forward pass
    outputs, hidden_states = rnn.forward(text_indices)
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output shape: {outputs[0].shape}")
    
    # Prediction
    predictions = rnn.predict(text_indices)
    predicted_chars = [idx_to_char[idx] for idx in predictions]
    print(f"Predicted sequence: {''.join(predicted_chars)}")
    
    # Simple LSTM demonstration
    print("\n2. Simple LSTM:")
    lstm = SimpleLSTM(input_size=vocab_size, hidden_size=20, output_size=vocab_size)
    
    # Forward pass
    outputs, hidden_states, cell_states = lstm.forward(text_indices)
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output shape: {outputs[0].shape}")
    print(f"Number of hidden states: {len(hidden_states)}")
    print(f"Number of cell states: {len(cell_states)}")
    
    # Prediction
    predictions = lstm.predict(text_indices)
    predicted_chars = [idx_to_char[idx] for idx in predictions]
    print(f"Predicted sequence: {''.join(predicted_chars)}")
    
    # Sequence-to-sequence model
    print("\n3. Sequence-to-Sequence Model:")
    seq2seq = SequenceToSequence(vocab_size=vocab_size, hidden_size=16)
    
    # Encode
    encoder_hidden, encoder_cell = seq2seq.encode(text_indices)
    print(f"Encoder hidden state shape: {encoder_hidden.shape}")
    print(f"Encoder cell state shape: {encoder_cell.shape}")
    
    # Decode
    outputs = seq2seq.decode(encoder_hidden, encoder_cell)
    print(f"Decoder outputs length: {len(outputs)}")
    
    # Text generation
    print("\n4. Text Generation:")
    generator = TextGenerator(lstm, idx_to_char, char_to_idx)
    
    seed = "hello"
    generated = generator.generate_text(seed, length=20, temperature=0.8)
    print(f"Seed text: '{seed}'")
    print(f"Generated text: '{generated}'")
    
    # Compare RNN and LSTM
    print("\n" + "="*50)
    print("Comparison: RNN vs LSTM")
    print("="*50)
    print("RNN:")
    print("- Simple recurrent connections")
    print("- Suffers from vanishing gradient problem")
    print("- Limited long-term memory")
    print("- Faster to train")
    
    print("\nLSTM:")
    print("- Gated architecture with input/forget/output gates")
    print("- Addresses vanishing gradient problem")
    print("- Better long-term memory retention")
    print("- More complex but more powerful")
    
    # Advanced sequence modeling concepts
    print("\n" + "="*50)
    print("Advanced Sequence Modeling Concepts")
    print("="*50)
    print("1. Attention Mechanisms:")
    print("   - Allow model to focus on relevant parts of input")
    print("   - Essential for sequence-to-sequence tasks")
    print("   - Basis for transformer models")
    
    print("\n2. Bidirectional RNNs:")
    print("   - Process sequences in both directions")
    print("   - Capture context from past and future")
    print("   - Useful for tasks like NER and POS tagging")
    
    print("\n3. GRU (Gated Recurrent Unit):")
    print("   - Simplified version of LSTM")
    print("   - Fewer parameters than LSTM")
    print("   - Similar performance in many tasks")
    
    print("\n4. Transformer Models:")
    print("   - Self-attention mechanisms")
    print("   - Parallel processing (no recurrence)")
    print("   - State-of-the-art performance")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of Sequence Models")
    print("="*50)
    print("1. Language Modeling:")
    print("   - Predict next word in sequence")
    print("   - Text generation")
    
    print("\n2. Machine Translation:")
    print("   - Translate text between languages")
    print("   - Encoder-decoder architectures")
    
    print("\n3. Named Entity Recognition:")
    print("   - Identify entities in text")
    print("   - Persons, organizations, locations")
    
    print("\n4. Part-of-Speech Tagging:")
    print("   - Label words with grammatical roles")
    print("   - Nouns, verbs, adjectives, etc.")
    
    print("\n5. Sentiment Analysis:")
    print("   - Analyze sentiment in sequences")
    print("   - Document-level or aspect-based")
    
    print("\n6. Speech Recognition:")
    print("   - Convert speech to text")
    print("   - Acoustic and language modeling")
    
    # Training considerations
    print("\n" + "="*50)
    print("Training Considerations")
    print("="*50)
    print("1. Gradient Issues:")
    print("   - Vanishing gradients in deep RNNs")
    print("   - Exploding gradients")
    print("   - Gradient clipping techniques")
    
    print("\n2. Optimization:")
    print("   - Adam optimizer often works well")
    print("   - Learning rate scheduling")
    print("   - Batch normalization for deep networks")
    
    print("\n3. Regularization:")
    print("   - Dropout for recurrent connections")
    print("   - Early stopping")
    print("   - Weight decay")
    
    print("\n4. Sequence Length:")
    print("   - Padding for variable-length sequences")
    print("   - Truncation for very long sequences")
    print("   - Memory considerations")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Sequence Models")
    print("="*50)
    print("1. Data Preprocessing:")
    print("   - Consistent tokenization")
    print("   - Appropriate sequence length")
    print("   - Handling of rare words/OOV tokens")
    
    print("\n2. Model Architecture:")
    print("   - Choose appropriate model complexity")
    print("   - Consider bidirectional models when context matters")
    print("   - Use pre-trained embeddings when available")
    
    print("\n3. Training Strategy:")
    print("   - Monitor for overfitting")
    print("   - Use validation sets for hyperparameter tuning")
    print("   - Implement proper evaluation metrics")
    
    print("\n4. Computational Efficiency:")
    print("   - Use GPU acceleration when possible")
    print("   - Consider model compression techniques")
    print("   - Batch processing for inference")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow/Keras: tf.keras.layers.LSTM, tf.keras.layers.GRU")
    print("- PyTorch: torch.nn.LSTM, torch.nn.GRU")
    print("- These provide GPU acceleration and optimized implementations")