"""
Transformers for NLP
===================

This module demonstrates transformer architectures for NLP tasks including BERT, GPT, and other transformer models.
It covers attention mechanisms, pre-training, and fine-tuning techniques.

Key Concepts:
- Self-Attention Mechanisms
- Transformer Encoder-Decoder Architecture
- BERT-style Models
- GPT-style Models
- Fine-tuning Transformers
"""

import numpy as np
from collections import defaultdict, Counter
import math


class SelfAttention:
    """
    Self-attention mechanism implementation.
    
    Parameters:
    -----------
    d_model : int
        Dimension of model
    num_heads : int, default=1
        Number of attention heads
    """
    
    def __init__(self, d_model, num_heads=1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize projection matrices
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
            Split tensor
        """
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Parameters:
        -----------
        Q : numpy array
            Query matrix
        K : numpy array
            Key matrix
        V : numpy array
            Value matrix
        mask : numpy array, optional
            Attention mask
            
        Returns:
        --------
        output : numpy array
            Attention output
        attention_weights : numpy array
            Attention weights
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Apply softmax
        attention_weights = self.softmax(scores)
        
        # Compute weighted sum
        output = np.matmul(attention_weights, V)
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x, mask=None):
        """
        Forward pass through self-attention.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, seq_len, d_model)
            Input tensor
        mask : numpy array, optional
            Attention mask
            
        Returns:
        --------
        output : numpy array
            Attention output
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Split heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(0, 2, 1, 3)
        concat_attention = attention_output.reshape(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = np.matmul(concat_attention, self.W_o)
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
    dropout_rate : float, default=0.1
        Dropout rate
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Sub-layers
        self.attention = SelfAttention(d_model, num_heads)
        self.feed_forward = self.create_feed_forward()
        
        # Layer normalization parameters
        self.layer_norm1_scale = np.ones(d_model)
        self.layer_norm1_bias = np.zeros(d_model)
        self.layer_norm2_scale = np.ones(d_model)
        self.layer_norm2_bias = np.zeros(d_model)
    
    def create_feed_forward(self):
        """Create feed-forward network."""
        # Simplified feed-forward network
        class FeedForward:
            def __init__(self, d_model, d_ff):
                self.W1 = np.random.randn(d_model, d_ff) * 0.1
                self.W2 = np.random.randn(d_ff, d_model) * 0.1
                self.b1 = np.zeros((1, d_ff))
                self.b2 = np.zeros((1, d_model))
            
            def forward(self, x):
                # First linear layer + ReLU
                hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)
                # Second linear layer
                output = np.matmul(hidden, self.W2) + self.b2
                return output
        
        return FeedForward(self.d_model, self.d_ff)
    
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
        epsilon : float, default=1e-6
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
    
    def forward(self, x, mask=None):
        """
        Forward pass through transformer encoder layer.
        
        Parameters:
        -----------
        x : numpy array
            Input tensor
        mask : numpy array, optional
            Attention mask
            
        Returns:
        --------
        output : numpy array
            Layer output
        """
        # Multi-head attention
        attn_output = self.attention.forward(x, mask)
        
        # Add & Norm
        out1 = self.layer_norm(x + attn_output, self.layer_norm1_scale, self.layer_norm1_bias)
        
        # Feed forward
        ff_output = self.feed_forward.forward(out1)
        
        # Add & Norm
        output = self.layer_norm(out1 + ff_output, self.layer_norm2_scale, self.layer_norm2_bias)
        
        return output


class BERTModel:
    """
    Simplified BERT model implementation.
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    d_model : int, default=768
        Dimension of model
    num_layers : int, default=12
        Number of transformer layers
    num_heads : int, default=12
        Number of attention heads
    d_ff : int, default=3072
        Dimension of feed-forward networks
    max_seq_length : int, default=512
        Maximum sequence length
    """
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, 
                 d_ff=3072, max_seq_length=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.1
        
        # Position embeddings
        self.position_embeddings = np.random.randn(max_seq_length, d_model) * 0.1
        
        # Segment embeddings (for sentence pairs)
        self.segment_embeddings = np.random.randn(2, d_model) * 0.1
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final layer normalization
        self.layer_norm_scale = np.ones(d_model)
        self.layer_norm_bias = np.zeros(d_model)
    
    def create_attention_mask(self, input_ids, attention_mask=None):
        """
        Create attention mask for padding tokens.
        
        Parameters:
        -----------
        input_ids : numpy array
            Input token IDs
        attention_mask : numpy array, optional
            Custom attention mask
            
        Returns:
        --------
        mask : numpy array
            Attention mask
        """
        if attention_mask is not None:
            return attention_mask
        
        # Create mask for padding tokens (assuming 0 is padding)
        mask = (input_ids != 0).astype(float)
        # Expand for attention computation
        mask = mask[:, np.newaxis, np.newaxis, :]
        return mask
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Forward pass through BERT model.
        
        Parameters:
        -----------
        input_ids : numpy array
            Input token IDs
        token_type_ids : numpy array, optional
            Token type IDs (for sentence pairs)
        attention_mask : numpy array, optional
            Attention mask
            
        Returns:
        --------
        sequence_output : numpy array
            Final sequence representations
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embeddings[input_ids]
        
        # Position embeddings
        position_ids = np.arange(seq_length)
        position_embeds = self.position_embeddings[position_ids]
        position_embeds = position_embeds[np.newaxis, :, :]  # Add batch dimension
        
        # Segment embeddings
        if token_type_ids is not None:
            segment_embeds = self.segment_embeddings[token_type_ids]
        else:
            segment_embeds = np.zeros_like(token_embeds)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds + segment_embeds
        
        # Apply transformer encoder layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer.forward(hidden_states)
        
        # Final layer normalization
        # Simplified normalization for demonstration
        sequence_output = hidden_states
        
        return sequence_output


class GPTModel:
    """
    Simplified GPT model implementation.
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    d_model : int, default=768
        Dimension of model
    num_layers : int, default=12
        Number of transformer layers
    num_heads : int, default=12
        Number of attention heads
    d_ff : int, default=3072
        Dimension of feed-forward networks
    max_seq_length : int, default=1024
        Maximum sequence length
    """
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12,
                 d_ff=3072, max_seq_length=1024):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, d_model) * 0.1
        
        # Position embeddings
        self.position_embeddings = np.random.randn(max_seq_length, d_model) * 0.1
        
        # Transformer decoder layers
        # Note: Simplified as encoder layers for demonstration
        self.decoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.1
    
    def forward(self, input_ids):
        """
        Forward pass through GPT model.
        
        Parameters:
        -----------
        input_ids : numpy array
            Input token IDs
            
        Returns:
        --------
        logits : numpy array
            Output logits
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embeddings[input_ids]
        
        # Position embeddings
        position_ids = np.arange(seq_length)
        position_embeds = self.position_embeddings[position_ids]
        position_embeds = position_embeds[np.newaxis, :, :]  # Add batch dimension
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        # Apply transformer decoder layers
        hidden_states = embeddings
        for layer in self.decoder_layers:
            # Causal mask for autoregressive modeling
            causal_mask = self.create_causal_mask(seq_length)
            hidden_states = layer.forward(hidden_states, causal_mask)
        
        # Output projection
        logits = np.matmul(hidden_states, self.lm_head)
        return logits
    
    def create_causal_mask(self, seq_length):
        """
        Create causal attention mask for autoregressive modeling.
        
        Parameters:
        -----------
        seq_length : int
            Sequence length
            
        Returns:
        --------
        mask : numpy array
            Causal mask
        """
        mask = np.tril(np.ones((seq_length, seq_length)))
        mask = mask[np.newaxis, np.newaxis, :, :]
        return mask


class TransformerFineTuner:
    """
    Fine-tuning utilities for transformer models.
    
    Parameters:
    -----------
    model : object
        Pre-trained transformer model
    num_classes : int, optional
        Number of classes for classification tasks
    """
    
    def __init__(self, model, num_classes=None):
        self.model = model
        self.num_classes = num_classes
        
        # Classification head (if needed)
        if num_classes:
            self.classifier = np.random.randn(model.d_model, num_classes) * 0.1
            self.classifier_bias = np.zeros((1, num_classes))
    
    def fine_tune_classification(self, input_ids, labels, learning_rate=1e-5, epochs=3):
        """
        Fine-tune model for classification task.
        
        Parameters:
        -----------
        input_ids : numpy array
            Input token IDs
        labels : numpy array
            Target labels
        learning_rate : float, default=1e-5
            Learning rate
        epochs : int, default=3
            Number of epochs
            
        Returns:
        --------
        losses : list
            Training losses
        """
        print("Fine-tuning transformer for classification...")
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Forward pass
            sequence_output = self.model.forward(input_ids)
            
            # Use [CLS] token representation (first token)
            cls_representation = sequence_output[:, 0, :]  # [batch_size, d_model]
            
            # Classification
            logits = np.matmul(cls_representation, self.classifier) + self.classifier_bias
            predictions = self.softmax(logits)
            
            # Compute loss
            loss = -np.mean(np.sum(labels * np.log(predictions + 1e-8), axis=1))
            total_loss += loss
            
            # Simplified backward pass (in practice, would compute gradients)
            # For demonstration, we'll just print progress
            
            avg_loss = total_loss / len(input_ids)
            losses.append(avg_loss)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Fine-tuning completed")
        return losses
    
    def softmax(self, x):
        """Softmax activation function."""
        # Subtract max for numerical stability
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Example usage and demonstration
if __name__ == "__main__":
    # Sample data for demonstration
    print("Transformers for NLP Demonstration")
    print("=" * 50)
    
    # Sample vocabulary
    vocab = ['[PAD]', '[CLS]', '[SEP]', 'hello', 'world', 'this', 'is', 'a', 'test', '[MASK]']
    vocab_size = len(vocab)
    word_to_id = {word: i for i, word in enumerate(vocab)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {vocab}")
    
    # Sample input sequences
    sample_input = np.array([
        [word_to_id['[CLS]'], word_to_id['hello'], word_to_id['world'], word_to_id['[SEP]']],
        [word_to_id['[CLS]'], word_to_id['this'], word_to_id['is'], word_to_id['a']]
    ])
    
    print(f"\nSample input shape: {sample_input.shape}")
    print(f"Sample input 1: {[id_to_word[i] for i in sample_input[0]]}")
    print(f"Sample input 2: {[id_to_word[i] for i in sample_input[1]]}")
    
    # Self-Attention demonstration
    print("\n1. Self-Attention Mechanism:")
    attention = SelfAttention(d_model=64, num_heads=8)
    
    # Sample input (batch_size=2, seq_len=4, d_model=64)
    sample_hidden = np.random.randn(2, 4, 64)
    attention_output = attention.forward(sample_hidden)
    
    print(f"Input shape: {sample_hidden.shape}")
    print(f"Output shape: {attention_output.shape}")
    
    # BERT Model demonstration
    print("\n2. BERT Model:")
    bert = BERTModel(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
    
    # Forward pass
    bert_output = bert.forward(sample_input)
    print(f"BERT input shape: {sample_input.shape}")
    print(f"BERT output shape: {bert_output.shape}")
    
    # GPT Model demonstration
    print("\n3. GPT Model:")
    gpt = GPTModel(vocab_size=vocab_size, d_model=128, num_layers=2, num_heads=4)
    
    # Forward pass
    gpt_logits = gpt.forward(sample_input)
    print(f"GPT input shape: {sample_input.shape}")
    print(f"GPT logits shape: {gpt_logits.shape}")
    
    # Fine-tuning demonstration
    print("\n4. Transformer Fine-tuning:")
    
    # Sample classification data
    num_classes = 3
    sample_labels = np.array([
        [1, 0, 0],  # Class 0
        [0, 1, 0]   # Class 1
    ])
    
    # Fine-tune BERT for classification
    finetuner = TransformerFineTuner(bert, num_classes=num_classes)
    losses = finetuner.fine_tune_classification(sample_input, sample_labels, epochs=2)
    print(f"Fine-tuning losses: {[f'{loss:.4f}' for loss in losses]}")
    
    # Compare BERT and GPT
    print("\n" + "="*50)
    print("Comparison: BERT vs GPT")
    print("="*50)
    print("BERT (Bidirectional Encoder Representations):")
    print("- Bidirectional attention (sees context from both directions)")
    print("- Encoder-only architecture")
    print("- Good for understanding tasks (classification, NER, QA)")
    print("- Uses [CLS] token for classification tasks")
    
    print("\nGPT (Generative Pre-trained Transformer):")
    print("- Autoregressive (causal) attention (sees context from left)")
    print("- Decoder-only architecture")
    print("- Good for generation tasks (text completion, dialogue)")
    print("- Uses next-token prediction for training")
    
    # Attention mechanisms
    print("\n" + "="*50)
    print("Attention Mechanisms")
    print("="*50)
    print("1. Self-Attention:")
    print("   - Computes attention scores between all positions")
    print("   - Allows model to focus on relevant parts")
    print("   - Parallelizable computation")
    
    print("\n2. Multi-Head Attention:")
    print("   - Multiple attention heads in parallel")
    print("   - Each head learns different representations")
    print("   - Concatenated and linearly transformed")
    
    print("\n3. Masked Attention:")
    print("   - Prevents attending to future positions")
    print("   - Essential for autoregressive models")
    print("   - Used in GPT-style models")
    
    # Pre-training tasks
    print("\n" + "="*50)
    print("Pre-training Tasks")
    print("="*50)
    print("1. BERT Pre-training:")
    print("   - Masked Language Modeling (MLM)")
    print("   - Next Sentence Prediction (NSP)")
    print("   - Learns bidirectional context")
    
    print("\n2. GPT Pre-training:")
    print("   - Causal Language Modeling")
    print("   - Next-token prediction")
    print("   - Learns left-to-right context")
    
    print("\n3. T5 Pre-training:")
    print("   - Text-to-Text framework")
    print("   - Various text transformation tasks")
    print("   - Unified approach to NLP tasks")
    
    # Fine-tuning strategies
    print("\n" + "="*50)
    print("Fine-tuning Strategies")
    print("="*50)
    print("1. Full Fine-tuning:")
    print("   - Update all model parameters")
    print("   - Best performance but computationally expensive")
    
    print("\n2. Layer-wise Learning Rates:")
    print("   - Lower rates for early layers")
    print("   - Higher rates for later layers")
    print("   - Better gradient flow")
    
    print("\n3. Adapter Modules:")
    print("   - Insert small trainable modules")
    print("   - Keep pre-trained weights frozen")
    print("   - Parameter-efficient fine-tuning")
    
    print("\n4. Prompt Tuning:")
    print("   - Learnable prompt tokens")
    print("   - Keep model weights frozen")
    print("   - Minimal parameter adaptation")
    
    # Applications
    print("\n" + "="*50)
    print("Transformer Applications")
    print("="*50)
    print("1. Text Classification:")
    print("   - Sentiment analysis")
    print("   - Document categorization")
    print("   - Intent classification")
    
    print("\n2. Question Answering:")
    print("   - Extractive QA")
    print("   - Generative QA")
    print("   - Open-domain QA")
    
    print("\n3. Named Entity Recognition:")
    print("   - Entity extraction")
    print("   - Entity linking")
    print("   - Information extraction")
    
    print("\n4. Machine Translation:")
    print("   - Neural machine translation")
    print("   - Multilingual models")
    print("   - Zero-shot translation")
    
    print("\n5. Text Generation:")
    print("   - Story generation")
    print("   - Code generation")
    print("   - Creative writing")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Transformers")
    print("="*50)
    print("1. Model Selection:")
    print("   - Choose appropriate model size")
    print("   - Consider domain-specific models")
    print("   - Balance performance and efficiency")
    
    print("\n2. Data Preprocessing:")
    print("   - Use appropriate tokenization")
    print("   - Handle special tokens correctly")
    print("   - Apply consistent preprocessing")
    
    print("\n3. Training Strategy:")
    print("   - Use learning rate scheduling")
    print("   - Implement gradient clipping")
    print("   - Monitor for overfitting")
    
    print("\n4. Fine-tuning:")
    print("   - Start with lower learning rates")
    print("   - Use appropriate batch sizes")
    print("   - Implement early stopping")
    
    print("\n5. Evaluation:")
    print("   - Use task-specific metrics")
    print("   - Validate on held-out data")
    print("   - Consider multiple evaluation aspects")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- Hugging Face Transformers: Pre-trained models and tokenizers")
    print("- TensorFlow/Keras: tf.keras.layers.Transformer")
    print("- PyTorch: torch.nn.Transformer")
    print("- These provide GPU acceleration and optimized implementations")