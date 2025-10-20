"""
Word Embeddings Implementation
=============================

This module demonstrates word embedding techniques including Word2Vec and GloVe.
It covers the theory, implementation, and applications of distributed word representations.

Key Concepts:
- Distributed Word Representations
- Word2Vec (Skip-gram and CBOW)
- GloVe (Global Vectors)
- Embedding Visualization
- Semantic Similarity
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random


class SimpleWord2Vec:
    """
    A simplified implementation of Word2Vec (Skip-gram model).
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of word embeddings
    learning_rate : float, default=0.01
        Learning rate for training
    negative_samples : int, default=5
        Number of negative samples
    """
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01, negative_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        
        # Initialize word embeddings (input vectors) and context embeddings (output vectors)
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Vocabulary mapping
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def build_vocab(self, sentences):
        """
        Build vocabulary from sentences.
        
        Parameters:
        -----------
        sentences : list
            List of tokenized sentences
        """
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Create vocabulary (most frequent words)
        vocab_words = [word for word, _ in word_counts.most_common(self.vocab_size)]
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def get_context_pairs(self, sentence, window_size=2):
        """
        Generate context pairs from a sentence.
        
        Parameters:
        -----------
        sentence : list
            Tokenized sentence
        window_size : int, default=2
            Context window size
            
        Returns:
        --------
        pairs : list
            List of (center_word, context_word) pairs
        """
        pairs = []
        sentence_indices = [self.word_to_idx.get(word, -1) for word in sentence]
        
        for i, center_idx in enumerate(sentence_indices):
            if center_idx == -1:  # Word not in vocabulary
                continue
                
            # Get context words within window
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)
            
            for j in range(start, end):
                if i != j and sentence_indices[j] != -1:
                    pairs.append((center_idx, sentence_indices[j]))
        
        return pairs
    
    def negative_sampling(self, target_idx):
        """
        Generate negative samples.
        
        Parameters:
        -----------
        target_idx : int
            Target word index
            
        Returns:
        --------
        negative_samples : list
            List of negative sample indices
        """
        # Simple uniform negative sampling (in practice, use unigram distribution)
        negative_samples = []
        while len(negative_samples) < self.negative_samples:
            neg_idx = random.randint(0, self.vocab_size - 1)
            if neg_idx != target_idx:
                negative_samples.append(neg_idx)
        return negative_samples
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def train(self, sentences, epochs=100, window_size=2):
        """
        Train the Word2Vec model.
        
        Parameters:
        -----------
        sentences : list
            List of tokenized sentences
        epochs : int, default=100
            Number of training epochs
        window_size : int, default=2
            Context window size
        """
        print("Training Word2Vec model...")
        
        # Build vocabulary
        self.build_vocab(sentences)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            total_pairs = 0
            
            for sentence in sentences:
                # Get context pairs
                pairs = self.get_context_pairs(sentence, window_size)
                
                for center_idx, context_idx in pairs:
                    # Positive sample
                    pos_score = np.dot(self.word_embeddings[center_idx], 
                                     self.context_embeddings[context_idx])
                    pos_prob = self.sigmoid(pos_score)
                    pos_loss = -np.log(pos_prob + 1e-8)
                    
                    # Negative samples
                    neg_samples = self.negative_sampling(context_idx)
                    neg_loss = 0
                    for neg_idx in neg_samples:
                        neg_score = np.dot(self.word_embeddings[center_idx], 
                                         self.context_embeddings[neg_idx])
                        neg_prob = self.sigmoid(-neg_score)
                        neg_loss += -np.log(neg_prob + 1e-8)
                    
                    # Total loss
                    loss = pos_loss + neg_loss
                    total_loss += loss
                    total_pairs += 1
                    
                    # Update embeddings (simplified gradient descent)
                    # In practice, would compute proper gradients
                    if pos_prob < 0.9:  # Only update if not confident enough
                        # Update positive sample
                        self.word_embeddings[center_idx] += self.learning_rate * (
                            1 - pos_prob) * self.context_embeddings[context_idx]
                        self.context_embeddings[context_idx] += self.learning_rate * (
                            1 - pos_prob) * self.word_embeddings[center_idx]
                        
                        # Update negative samples
                        for neg_idx in neg_samples:
                            neg_prob = self.sigmoid(np.dot(self.word_embeddings[center_idx], 
                                                         self.context_embeddings[neg_idx]))
                            self.word_embeddings[center_idx] -= self.learning_rate * (
                                neg_prob) * self.context_embeddings[neg_idx]
                            self.context_embeddings[neg_idx] -= self.learning_rate * (
                                neg_prob) * self.word_embeddings[center_idx]
            
            # Print progress
            if epoch % 20 == 0 and total_pairs > 0:
                avg_loss = total_loss / total_pairs
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("Word2Vec training completed")
    
    def get_embedding(self, word):
        """
        Get embedding for a word.
        
        Parameters:
        -----------
        word : str
            Input word
            
        Returns:
        --------
        embedding : numpy array
            Word embedding vector
        """
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.word_embeddings[idx]
        else:
            raise ValueError(f"Word '{word}' not in vocabulary")
    
    def similarity(self, word1, word2):
        """
        Compute cosine similarity between two words.
        
        Parameters:
        -----------
        word1 : str
            First word
        word2 : str
            Second word
            
        Returns:
        --------
        similarity : float
            Cosine similarity
        """
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def most_similar(self, word, top_k=5):
        """
        Find most similar words to a given word.
        
        Parameters:
        -----------
        word : str
            Input word
        top_k : int, default=5
            Number of similar words to return
            
        Returns:
        --------
        similar_words : list
            List of (word, similarity) tuples
        """
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_emb = self.get_embedding(word)
        similarities = []
        
        for other_word, other_idx in self.word_to_idx.items():
            if other_word != word:
                other_emb = self.word_embeddings[other_idx]
                sim = self.cosine_similarity(word_emb, other_emb)
                similarities.append((other_word, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


class SimpleGloVe:
    """
    A simplified implementation of GloVe (Global Vectors).
    
    Parameters:
    -----------
    vocab_size : int
        Size of vocabulary
    embedding_dim : int
        Dimension of word embeddings
    learning_rate : float, default=0.05
        Learning rate for training
    max_count : int, default=100
        Maximum co-occurrence count
    alpha : float, default=0.75
        Weighting parameter
    """
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.05, max_count=100, alpha=0.75):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.max_count = max_count
        self.alpha = alpha
        
        # Initialize word embeddings and context embeddings
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.context_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        # Initialize biases
        self.word_biases = np.zeros(vocab_size)
        self.context_biases = np.zeros(vocab_size)
        
        # Co-occurrence matrix
        self.cooccur = defaultdict(lambda: defaultdict(int))
        
        # Vocabulary mapping
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def build_vocab(self, sentences):
        """
        Build vocabulary from sentences.
        
        Parameters:
        -----------
        sentences : list
            List of tokenized sentences
        """
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Create vocabulary (most frequent words)
        vocab_words = [word for word, _ in word_counts.most_common(self.vocab_size)]
        
        # Create mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def build_cooccur_matrix(self, sentences, window_size=2):
        """
        Build co-occurrence matrix from sentences.
        
        Parameters:
        -----------
        sentences : list
            List of tokenized sentences
        window_size : int, default=2
            Context window size
        """
        print("Building co-occurrence matrix...")
        
        for sentence in sentences:
            sentence_indices = [self.word_to_idx.get(word, -1) for word in sentence]
            
            for i, center_idx in enumerate(sentence_indices):
                if center_idx == -1:  # Word not in vocabulary
                    continue
                    
                # Get context words within window
                start = max(0, i - window_size)
                end = min(len(sentence_indices), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j and sentence_indices[j] != -1:
                        context_idx = sentence_indices[j]
                        # Distance weighting (closer words have higher weights)
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        self.cooccur[center_idx][context_idx] += weight
        
        print(f"Co-occurrence matrix built with {len(self.cooccur)} entries")
    
    def compute_weight(self, count):
        """
        Compute GloVe weighting function.
        
        Parameters:
        -----------
        count : float
            Co-occurrence count
            
        Returns:
        --------
        weight : float
            Weight value
        """
        if count / self.max_count > 1:
            return 1
        else:
            return (count / self.max_count) ** self.alpha
    
    def train(self, sentences, epochs=100, window_size=2):
        """
        Train the GloVe model.
        
        Parameters:
        -----------
        sentences : list
            List of tokenized sentences
        epochs : int, default=100
            Number of training epochs
        window_size : int, default=2
            Context window size
        """
        print("Training GloVe model...")
        
        # Build vocabulary and co-occurrence matrix
        self.build_vocab(sentences)
        self.build_cooccur_matrix(sentences, window_size)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            total_entries = 0
            
            # Iterate through co-occurrence entries
            for i, (word_idx, context_dict) in enumerate(self.cooccur.items()):
                for context_idx, count in context_dict.items():
                    if count > 0:
                        # Compute prediction
                        emb_product = np.dot(self.word_embeddings[word_idx], 
                                           self.context_embeddings[context_idx])
                        pred = emb_product + self.word_biases[word_idx] + self.context_biases[context_idx]
                        
                        # Compute weight
                        weight = self.compute_weight(count)
                        
                        # Compute loss gradient
                        diff = pred - np.log(count)
                        gradient = weight * diff
                        
                        # Update embeddings and biases
                        # In practice, would use proper gradient descent
                        self.word_embeddings[word_idx] -= self.learning_rate * gradient * self.context_embeddings[context_idx]
                        self.context_embeddings[context_idx] -= self.learning_rate * gradient * self.word_embeddings[word_idx]
                        self.word_biases[word_idx] -= self.learning_rate * gradient
                        self.context_biases[context_idx] -= self.learning_rate * gradient
                        
                        # Accumulate loss
                        total_loss += weight * (diff ** 2)
                        total_entries += 1
            
            # Print progress
            if epoch % 20 == 0 and total_entries > 0:
                avg_loss = total_loss / total_entries
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        print("GloVe training completed")
    
    def get_embedding(self, word):
        """
        Get embedding for a word (combining word and context embeddings).
        
        Parameters:
        -----------
        word : str
            Input word
            
        Returns:
        --------
        embedding : numpy array
            Combined word embedding vector
        """
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            # Combine word and context embeddings
            return (self.word_embeddings[idx] + self.context_embeddings[idx]) / 2
        else:
            raise ValueError(f"Word '{word}' not in vocabulary")


# Example usage and demonstration
if __name__ == "__main__":
    # Sample corpus for demonstration
    corpus = [
        ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        ["a", "quick", "brown", "dog", "jumps", "over", "a", "lazy", "fox"],
        ["the", "lazy", "dog", "sleeps", "under", "the", "quick", "brown", "fox"],
        ["brown", "foxes", "are", "quick", "and", "lazy", "dogs", "sleep"],
        ["quick", "brown", "animals", "jump", "over", "lazy", "ones"],
        ["the", "dog", "is", "lazy", "but", "the", "fox", "is", "quick"],
        ["brown", "animals", "like", "foxes", "and", "dogs", "are", "common"],
        ["quick", "movements", "characterize", "the", "brown", "fox"],
        ["lazy", "behavior", "is", "typical", "for", "the", "dog"],
        ["foxes", "and", "dogs", "are", "both", "brown", "animals"]
    ]
    
    print("Word Embeddings Demonstration")
    print("=" * 50)
    
    # Word2Vec demonstration
    print("\n1. Word2Vec (Skip-gram Model):")
    word2vec = SimpleWord2Vec(vocab_size=20, embedding_dim=10, learning_rate=0.01)
    word2vec.train(corpus, epochs=50)
    
    # Test word similarities
    test_words = ["quick", "lazy", "brown", "fox", "dog"]
    print(f"\nSample word embeddings for vocabulary words:")
    for word in test_words:
        if word in word2vec.word_to_idx:
            embedding = word2vec.get_embedding(word)
            print(f"  {word}: {embedding[:5]}...")  # Show first 5 dimensions
    
    # Find similar words
    print(f"\nMost similar words:")
    for word in ["quick", "brown"]:
        if word in word2vec.word_to_idx:
            similar = word2vec.most_similar(word, top_k=3)
            print(f"  {word}: {similar}")
    
    # GloVe demonstration
    print("\n2. GloVe (Global Vectors):")
    glove = SimpleGloVe(vocab_size=20, embedding_dim=10, learning_rate=0.05)
    glove.train(corpus, epochs=50)
    
    # Test GloVe embeddings
    print(f"\nSample GloVe embeddings:")
    for word in test_words:
        if word in glove.word_to_idx:
            embedding = glove.get_embedding(word)
            print(f"  {word}: {embedding[:5]}...")  # Show first 5 dimensions
    
    # Compare Word2Vec and GloVe
    print("\n" + "="*50)
    print("Comparison: Word2Vec vs GloVe")
    print("="*50)
    print("Word2Vec:")
    print("- Predicts context words given a center word (or vice versa)")
    print("- Uses local context windows")
    print("- Trained with neural network and backpropagation")
    print("- Good for analogy tasks")
    
    print("\nGloVe:")
    print("- Factorizes word-word co-occurrence matrix")
    print("- Uses global statistical information")
    print("- Trained with least squares optimization")
    print("- Good for word similarity tasks")
    
    # Visualization concept (without actual plotting for simplicity)
    print("\n" + "="*50)
    print("Embedding Visualization Concept")
    print("="*50)
    print("Word embeddings can be visualized using techniques like:")
    print("- t-SNE (t-distributed Stochastic Neighbor Embedding)")
    print("- PCA (Principal Component Analysis)")
    print("- UMAP (Uniform Manifold Approximation and Projection)")
    print("\nThese techniques reduce high-dimensional embeddings to 2D/3D for visualization.")
    print("Semantically similar words appear close together in the visualization.")
    
    # Mathematical foundations
    print("\n" + "="*50)
    print("Mathematical Foundations")
    print("="*50)
    print("Word2Vec (Skip-gram):")
    print("  P(context|center) = softmax(center · context)")
    print("  Loss = -log P(context|center)")
    
    print("\nGloVe:")
    print("  J = Σᵢⱼ f(Xᵢⱼ)(wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ - log Xᵢⱼ)²")
    print("  Where f is the weighting function and Xᵢⱼ is co-occurrence count")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of Word Embeddings")
    print("="*50)
    print("1. Text Classification:")
    print("   - Sentiment analysis")
    print("   - Document categorization")
    
    print("\n2. Information Retrieval:")
    print("   - Document similarity")
    print("   - Search query expansion")
    
    print("\n3. Machine Translation:")
    print("   - Cross-lingual embeddings")
    print("   - Translation quality improvement")
    
    print("\n4. Question Answering:")
    print("   - Semantic similarity between questions and answers")
    print("   - Context understanding")
    
    print("\n5. Named Entity Recognition:")
    print("   - Entity type classification")
    print("   - Contextual entity understanding")
    
    # Popular pre-trained embeddings
    print("\n" + "="*50)
    print("Popular Pre-trained Embeddings")
    print("="*50)
    print("1. Word2Vec:")
    print("   - Google News Word2Vec (300 dimensions)")
    print("   - Trained on Google News dataset")
    
    print("\n2. GloVe:")
    print("   - GloVe Common Crawl (300 dimensions)")
    print("   - Trained on Common Crawl dataset")
    
    print("\n3. FastText:")
    print("   - Subword information")
    print("   - Better handling of rare words")
    
    print("\n4. Contextual Embeddings:")
    print("   - ELMo (Embeddings from Language Models)")
    print("   - BERT (Bidirectional Encoder Representations)")
    print("   - GPT (Generative Pre-trained Transformer)")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Word Embeddings")
    print("="*50)
    print("1. Domain Consideration:")
    print("   - Use domain-specific embeddings when available")
    print("   - Fine-tune pre-trained embeddings on domain data")
    
    print("\n2. Dimensionality:")
    print("   - Balance between expressiveness and computational cost")
    print("   - Typical range: 100-300 dimensions")
    
    print("\n3. Vocabulary Size:")
    print("   - Include most frequent words in vocabulary")
    print("   - Handle out-of-vocabulary words appropriately")
    
    print("\n4. Evaluation:")
    print("   - Intrinsic evaluation: Word similarity, analogy tasks")
    print("   - Extrinsic evaluation: Downstream task performance")
    
    print("\n5. Preprocessing:")
    print("   - Apply consistent text preprocessing")
    print("   - Consider lowercasing, stemming, etc.")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- Gensim: Word2Vec, FastText, Doc2Vec")
    print("- spaCy: Pre-trained embeddings")
    print("- TensorFlow/PyTorch: Custom embedding layers")
    print("- These provide GPU acceleration and optimized implementations")