"""
Transformers Implementation
==========================

This module demonstrates the implementation of Transformer architectures and attention mechanisms.
It covers self-attention, multi-head attention, and transformer encoder-decoder structures.

Key Concepts:
- Self-Attention Mechanism
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder-Decoder
- BERT-style Models
- GPT-style Models
"""

import numpy as np
import matplotlib.pyplot as plt


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Parameters:
    -----------
    Q : numpy array of shape (..., seq_len_q, d_k)
        Query matrix
    K : numpy array of shape (..., seq_len_k, d_k)
        Key matrix
    V : numpy array of shape (..., seq_len_k, d_v)
        Value matrix
    mask : numpy array, optional
        Mask to apply to attention scores
        
    Returns:
    --------
    output : numpy array of shape (..., seq_len_q, d_v)
        Attention output
    attention_weights : numpy array of shape (..., seq_len_q, seq_len_k)
        Attention weights
    """
    # Compute attention scores
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores += (mask * -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scores)
    
    # Compute weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax(x):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    """
    Multi-head attention mechanism.
    
    Parameters:
    -----------
    d_model : int
        Dimension of model
    num_heads : int
        Number of attention heads
    """
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        
        Parameters:
        -----------
        x : numpy array
            Input tensor
        batch_size : int
            Batch size
            
        Returns:
        --------
        x : numpy array
            Split tensor with shape (batch_size, num_heads, seq_len, d_k)
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass of multi-head attention.
        
        Parameters:
        -----------
        q : numpy array of shape (batch_size, seq_len_q, d_model)
            Query tensor
        k : numpy array of shape (batch_size, seq_len_k, d_model)
            Key tensor
        v : numpy array of shape (batch_size, seq_len_v, d_model)
            Value tensor
        mask : numpy array, optional
            Attention mask
            
        Returns:
        --------
        output : numpy array of shape (batch_size, seq_len_q, d_model)
            Multi-head attention output
        """
        batch_size = q.shape[0]
        
        # Linear projections
        Q = np.matmul(q, self.W_q)  # (batch_size, seq_len_q, d_model)
        K = np.matmul(k, self.W_k)  # (batch_size, seq_len_k, d_model)
        V = np.matmul(v, self.W_v)  # (batch_size, seq_len_v, d_model)
        
        # Split heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        concat_attention = attention_output.reshape(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = np.matmul(concat_attention, self.W_o)
        
        return output, attention_weights


class PositionalEncoding:
    """
    Positional encoding layer.
    
    Parameters:
    -----------
    d_model : int
        Dimension of model
    max_len : int
        Maximum sequence length
    """
    
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * 
                         -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe.reshape(1, max_len, d_model)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, seq_len, d_model)
            Input tensor
            
        Returns:
        --------
        x : numpy array of shape (batch_size, seq_len, d_model)
            Input with positional encoding added
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return x


class FeedForward:
    """
    Position-wise feed-forward network.
    
    Parameters:
    -----------
    d_model : int
        Dimension of model
    d_ff : int
        Dimension of feed-forward network
    """
    
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weights
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros((1, d_ff))
        self.b2 = np.zeros((1, d_model))
    
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, seq_len, d_model)
            Input tensor
            
        Returns:
        --------
        output : numpy array of shape (batch_size, seq_len, d_model)
            Output tensor
        """
        # First linear layer + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)
        
        # Second linear layer
        output = np.matmul(hidden, self.W2) + self.b2
        
        return output


class TransformerEncoderLayer:
    """
    Single transformer encoder layer.
    
    Parameters:
    -----------
    d_model : int
        Dimension of model
    num_heads : int
        Number of attention heads
    d_ff : int
        Dimension of feed-forward network
    dropout_rate : float
        Dropout rate
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Sub-layers
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer normalization parameters
        self.layernorm1_scale = np.ones(d_model)
        self.layernorm1_bias = np.zeros(d_model)
        self.layernorm2_scale = np.ones(d_model)
        self.layernorm2_bias = np.zeros(d_model)
    
    def layer_norm(self, x, scale, bias, epsilon=1e-6):
        """
        Apply layer normalization.
        
        Parameters:
        -----------
        x : numpy array
            Input tensor
        scale : numpy array
            Scale parameters
        bias : numpy array
            Bias parameters
        epsilon : float
            Small constant for numerical stability
            
        Returns:
        --------
        output : numpy array
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + epsilon)
        return scale * normalized + bias
    
    def forward(self, x, training=True):
        """
        Forward pass of transformer encoder layer.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, seq_len, d_model)
            Input tensor
        training : bool
            Whether in training mode
            
        Returns:
        --------
        output : numpy array of shape (batch_size, seq_len, d_model)
            Output tensor
        """
        # Multi-head attention
        attn_output, _ = self.mha.forward(x, x, x)
        
        # Dropout (simplified)
        if training and self.dropout_rate > 0:
            # In practice, apply dropout
            pass
        
        # Add & Norm
        out1 = self.layer_norm(x + attn_output, self.layernorm1_scale, self.layernorm1_bias)
        
        # Feed forward
        ffn_output = self.ffn.forward(out1)
        
        # Dropout (simplified)
        if training and self.dropout_rate > 0:
            # In practice, apply dropout
            pass
        
        # Add & Norm
        output = self.layer_norm(out1 + ffn_output, self.layernorm2_scale, self.layernorm2_bias)
        
        return output


class SimpleTransformer:
    """
    A simplified transformer model for educational purposes.
    
    Parameters:
    -----------
    num_layers : int
        Number of encoder/decoder layers
    d_model : int
        Dimension of model
    num_heads : int
        Number of attention heads
    d_ff : int
        Dimension of feed-forward networks
    input_vocab_size : int
        Size of input vocabulary
    target_vocab_size : int
        Size of target vocabulary
    max_len : int
        Maximum sequence length
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff,
                 input_vocab_size, target_vocab_size, max_len=512):
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_len = max_len
        
        # Embedding layers
        self.embedding_scale = np.sqrt(d_model)
        self.encoder_embedding = np.random.randn(input_vocab_size, d_model) * 0.1
        self.decoder_embedding = np.random.randn(target_vocab_size, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Decoder layers would go here in a full implementation
        
        # Final linear layer
        self.final_layer = np.random.randn(d_model, target_vocab_size) * 0.1


# Example usage and demonstration
if __name__ == "__main__":
    # Demonstrate attention mechanism
    print("Transformer Architecture Demonstration")
    print("="*50)
    
    # Create sample data for attention
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_k = 4
    
    # Sample query, key, value matrices
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    
    print("Scaled Dot-Product Attention:")
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show sample attention weights
    print(f"\nSample attention weights for first sequence:")
    print(attention_weights[0])
    
    # Demonstrate multi-head attention
    print("\n" + "="*50)
    print("Multi-Head Attention:")
    print("="*50)
    
    num_heads = 2
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Sample input sequences
    seq_q = np.random.randn(batch_size, seq_len, d_model)
    seq_k = np.random.randn(batch_size, seq_len, d_model)
    seq_v = np.random.randn(batch_size, seq_len, d_model)
    
    mha_output, mha_weights = mha.forward(seq_q, seq_k, seq_v)
    print(f"Multi-head attention input shape: {seq_q.shape}")
    print(f"Multi-head attention output shape: {mha_output.shape}")
    print(f"Attention weights shape: {mha_weights.shape}")
    
    # Demonstrate positional encoding
    print("\n" + "="*50)
    print("Positional Encoding:")
    print("="*50)
    
    pos_enc = PositionalEncoding(d_model, max_len=10)
    print(f"Positional encoding shape: {pos_enc.pe.shape}")
    
    # Show first few positions
    print("First 5 positions of positional encoding:")
    print(pos_enc.pe[0, :5, :])
    
    # Demonstrate with input
    sample_input = np.random.randn(1, 5, d_model)
    input_with_pos = pos_enc.forward(sample_input)
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Input with positional encoding shape: {input_with_pos.shape}")
    
    # Demonstrate feed-forward network
    print("\n" + "="*50)
    print("Feed-Forward Network:")
    print("="*50)
    
    d_ff = 16
    ffn = FeedForward(d_model, d_ff)
    
    ffn_output = ffn.forward(sample_input)
    print(f"Feed-forward input shape: {sample_input.shape}")
    print(f"Feed-forward output shape: {ffn_output.shape}")
    
    # Demonstrate transformer encoder layer
    print("\n" + "="*50)
    print("Transformer Encoder Layer:")
    print("="*50)
    
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    encoder_output = encoder_layer.forward(sample_input)
    print(f"Encoder layer input shape: {sample_input.shape}")
    print(f"Encoder layer output shape: {encoder_output.shape}")
    
    # Demonstrate full transformer
    print("\n" + "="*50)
    print("Full Transformer Model:")
    print("="*50)
    
    vocab_size = 1000
    transformer = SimpleTransformer(
        num_layers=2,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        max_len=512
    )
    
    print(f"Transformer model created with {transformer.num_layers} layers")
    print(f"Model dimension: {transformer.d_model}")
    print(f"Number of heads: {transformer.num_heads}")
    print(f"Input vocabulary size: {transformer.input_vocab_size}")
    print(f"Target vocabulary size: {transformer.target_vocab_size}")
    
    # Key concepts explanation
    print("\n" + "="*50)
    print("Key Transformer Concepts:")
    print("="*50)
    print("1. Self-Attention: Allows model to focus on different parts of input")
    print("2. Multi-Head Attention: Parallel attention layers for different representations")
    print("3. Positional Encoding: Adds sequence order information to input embeddings")
    print("4. Residual Connections: Help with gradient flow during training")
    print("5. Layer Normalization: Stabilizes training")
    print("6. Encoder-Decoder Architecture: Processes input and generates output sequences")
    
    # Popular transformer models
    print("\n" + "="*50)
    print("Popular Transformer Models:")
    print("="*50)
    print("1. BERT: Bidirectional Encoder Representations from Transformers")
    print("2. GPT: Generative Pre-trained Transformer")
    print("3. T5: Text-to-Text Transfer Transformer")
    print("4. Transformer-XL: Long sequence modeling")
    print("5. BART: Denoising autoencoder for text generation")
    
    # Applications
    print("\n" + "="*50)
    print("Transformer Applications:")
    print("="*50)
    print("- Machine Translation")
    print("- Text Summarization")
    print("- Question Answering")
    print("- Text Generation")
    print("- Named Entity Recognition")
    print("- Sentiment Analysis")
    print("- Code Generation")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- Hugging Face Transformers")
    print("- TensorFlow/Keras")
    print("- PyTorch")
    print("- These provide pre-trained models and optimized implementations")