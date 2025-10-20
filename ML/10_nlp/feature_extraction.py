"""
Feature Extraction for NLP
=========================

This module demonstrates various feature extraction techniques for NLP tasks.
It covers Bag of Words, TF-IDF, n-grams, and other text representation methods.

Key Concepts:
- Bag of Words (BoW) Representation
- Term Frequency-Inverse Document Frequency (TF-IDF)
- N-gram Models
- Text Vectorization
- Feature Selection
"""

import numpy as np
from collections import Counter, defaultdict
import math


class BagOfWords:
    """
    Bag of Words feature extractor.
    
    Parameters:
    -----------
    max_features : int, optional
        Maximum number of features to keep
    min_df : int, default=1
        Minimum document frequency for a term
    max_df : float, default=1.0
        Maximum document frequency for a term (proportion)
    lowercase : bool, default=True
        Whether to convert text to lowercase
    """
    
    def __init__(self, max_features=None, min_df=1, max_df=1.0, lowercase=True):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.vocabulary = {}
        self.feature_names = []
        
    def tokenize(self, text):
        """
        Simple tokenization method.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        tokens : list
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        return text.split()
    
    def fit(self, documents):
        """
        Learn vocabulary from documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        self : BagOfWords
            Fitted vectorizer
        """
        # Count term frequencies in each document
        term_doc_freq = defaultdict(int)  # Document frequency for each term
        term_freq = defaultdict(int)      # Total frequency for each term
        
        # Process each document
        for doc in documents:
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)
            
            # Update document frequencies
            for token in unique_tokens:
                term_doc_freq[token] += 1
            
            # Update term frequencies
            for token in tokens:
                term_freq[token] += 1
        
        # Filter terms based on document frequency
        total_docs = len(documents)
        min_df_count = self.min_df
        max_df_count = self.max_df * total_docs if isinstance(self.max_df, float) else self.max_df
        
        # Build vocabulary
        vocab_items = []
        for term, doc_freq in term_doc_freq.items():
            if doc_freq >= min_df_count and doc_freq <= max_df_count:
                vocab_items.append((term, term_freq[term]))
        
        # Sort by frequency and limit features
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            vocab_items = vocab_items[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(vocab_items)}
        self.feature_names = [term for term, _ in vocab_items]
        
        return self
    
    def transform(self, documents):
        """
        Transform documents to document-term matrix.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            Document-term matrix
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            term_counts = Counter(tokens)
            
            for term, count in term_counts.items():
                if term in self.vocabulary:
                    feature_idx = self.vocabulary[term]
                    X[doc_idx, feature_idx] = count
        
        return X
    
    def fit_transform(self, documents):
        """
        Learn vocabulary and transform documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            Document-term matrix
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self):
        """
        Get feature names.
        
        Returns:
        --------
        feature_names : list
            List of feature names
        """
        return self.feature_names


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer.
    
    Parameters:
    -----------
    max_features : int, optional
        Maximum number of features to keep
    min_df : int, default=1
        Minimum document frequency for a term
    max_df : float, default=1.0
        Maximum document frequency for a term (proportion)
    lowercase : bool, default=True
        Whether to convert text to lowercase
    norm : str, default='l2'
        Normalization method ('l1', 'l2', or None)
    """
    
    def __init__(self, max_features=None, min_df=1, max_df=1.0, 
                 lowercase=True, norm='l2'):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.norm = norm
        self.vocabulary = {}
        self.feature_names = []
        self.idf_values = []
        
    def tokenize(self, text):
        """
        Simple tokenization method.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        tokens : list
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        return text.split()
    
    def fit(self, documents):
        """
        Learn vocabulary and IDF values from documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        self : TFIDFVectorizer
            Fitted vectorizer
        """
        # Count document frequencies
        term_doc_freq = defaultdict(int)
        total_docs = len(documents)
        
        # Process each document
        for doc in documents:
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)
            
            # Update document frequencies
            for token in unique_tokens:
                term_doc_freq[token] += 1
        
        # Filter terms based on document frequency
        min_df_count = self.min_df
        max_df_count = self.max_df * total_docs if isinstance(self.max_df, float) else self.max_df
        
        # Build vocabulary with IDF values
        vocab_items = []
        for term, doc_freq in term_doc_freq.items():
            if doc_freq >= min_df_count and doc_freq <= max_df_count:
                # Calculate IDF: log(N / df)
                idf = math.log(total_docs / doc_freq)
                vocab_items.append((term, idf))
        
        # Sort by IDF and limit features
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            vocab_items = vocab_items[:self.max_features]
        
        # Create vocabulary mapping and IDF values
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(vocab_items)}
        self.feature_names = [term for term, _ in vocab_items]
        self.idf_values = [idf for _, idf in vocab_items]
        
        return self
    
    def transform(self, documents):
        """
        Transform documents to TF-IDF matrix.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            TF-IDF matrix
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            term_counts = Counter(tokens)
            doc_length = len(tokens)
            
            # Calculate TF-IDF for each term
            for term, count in term_counts.items():
                if term in self.vocabulary:
                    feature_idx = self.vocabulary[term]
                    
                    # Term Frequency (normalized by document length)
                    tf = count / doc_length if doc_length > 0 else 0
                    
                    # Inverse Document Frequency
                    idf = self.idf_values[feature_idx]
                    
                    # TF-IDF
                    X[doc_idx, feature_idx] = tf * idf
        
        # Apply normalization
        if self.norm == 'l1':
            # L1 normalization (sum of absolute values = 1)
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X = X / norms
        elif self.norm == 'l2':
            # L2 normalization (Euclidean norm = 1)
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
            norms[norms == 0] = 1  # Avoid division by zero
            X = X / norms
        
        return X
    
    def fit_transform(self, documents):
        """
        Learn vocabulary and IDF values, then transform documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            TF-IDF matrix
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self):
        """
        Get feature names.
        
        Returns:
        --------
        feature_names : list
            List of feature names
        """
        return self.feature_names


class NGramVectorizer:
    """
    N-gram feature extractor.
    
    Parameters:
    -----------
    n : int, default=2
        Size of n-grams (2 for bigrams, 3 for trigrams, etc.)
    max_features : int, optional
        Maximum number of features to keep
    min_df : int, default=1
        Minimum document frequency for a term
    lowercase : bool, default=True
        Whether to convert text to lowercase
    """
    
    def __init__(self, n=2, max_features=None, min_df=1, lowercase=True):
        self.n = n
        self.max_features = max_features
        self.min_df = min_df
        self.lowercase = lowercase
        self.vocabulary = {}
        self.feature_names = []
        
    def generate_ngrams(self, tokens, n):
        """
        Generate n-grams from tokens.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
        n : int
            Size of n-grams
            
        Returns:
        --------
        ngrams : list
            List of n-grams
        """
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def tokenize(self, text):
        """
        Simple tokenization method.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        tokens : list
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        return text.split()
    
    def fit(self, documents):
        """
        Learn vocabulary from documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        self : NGramVectorizer
            Fitted vectorizer
        """
        # Count n-gram frequencies
        ngram_doc_freq = defaultdict(int)  # Document frequency for each n-gram
        ngram_freq = defaultdict(int)      # Total frequency for each n-gram
        
        # Process each document
        for doc in documents:
            tokens = self.tokenize(doc)
            ngrams = self.generate_ngrams(tokens, self.n)
            unique_ngrams = set(ngrams)
            
            # Update document frequencies
            for ngram in unique_ngrams:
                ngram_doc_freq[ngram] += 1
            
            # Update n-gram frequencies
            for ngram in ngrams:
                ngram_freq[ngram] += 1
        
        # Filter n-grams based on document frequency
        min_df_count = self.min_df
        
        # Build vocabulary
        vocab_items = []
        for ngram, doc_freq in ngram_doc_freq.items():
            if doc_freq >= min_df_count:
                vocab_items.append((ngram, ngram_freq[ngram]))
        
        # Sort by frequency and limit features
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        if self.max_features:
            vocab_items = vocab_items[:self.max_features]
        
        # Create vocabulary mapping
        self.vocabulary = {ngram: idx for idx, (ngram, _) in enumerate(vocab_items)}
        self.feature_names = [ngram for ngram, _ in vocab_items]
        
        return self
    
    def transform(self, documents):
        """
        Transform documents to n-gram matrix.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            N-gram matrix
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        X = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self.tokenize(doc)
            ngrams = self.generate_ngrams(tokens, self.n)
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if ngram in self.vocabulary:
                    feature_idx = self.vocabulary[ngram]
                    X[doc_idx, feature_idx] = count
        
        return X
    
    def fit_transform(self, documents):
        """
        Learn vocabulary and transform documents.
        
        Parameters:
        -----------
        documents : list
            List of text documents
            
        Returns:
        --------
        X : numpy array of shape (n_documents, n_features)
            N-gram matrix
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self):
        """
        Get feature names.
        
        Returns:
        --------
        feature_names : list
            List of feature names
        """
        return self.feature_names


# Example usage and demonstration
if __name__ == "__main__":
    # Sample documents for demonstration
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog jumps over a lazy fox",
        "The lazy dog sleeps under the quick brown fox",
        "Brown foxes are quick and lazy dogs sleep",
        "Quick brown animals jump over lazy ones"
    ]
    
    print("Feature Extraction for NLP Demonstration")
    print("=" * 50)
    
    # Bag of Words
    print("\n1. Bag of Words (BoW):")
    bow = BagOfWords(max_features=10, min_df=1)
    bow_matrix = bow.fit_transform(documents)
    
    print(f"Vocabulary size: {len(bow.vocabulary)}")
    print(f"Feature names: {bow.get_feature_names()}")
    print(f"Matrix shape: {bow_matrix.shape}")
    print("Sample BoW matrix (first 3 documents, first 5 features):")
    print(bow_matrix[:3, :5])
    
    # TF-IDF
    print("\n2. TF-IDF Vectorization:")
    tfidf = TFIDFVectorizer(max_features=10, min_df=1)
    tfidf_matrix = tfidf.fit_transform(documents)
    
    print(f"Vocabulary size: {len(tfidf.vocabulary)}")
    print(f"Feature names: {tfidf.get_feature_names()}")
    print(f"Matrix shape: {tfidf_matrix.shape}")
    print("Sample TF-IDF matrix (first 3 documents, first 5 features):")
    print(tfidf_matrix[:3, :5])
    
    # N-grams
    print("\n3. N-gram Vectorization:")
    
    # Bigrams
    bigram = NGramVectorizer(n=2, max_features=10, min_df=1)
    bigram_matrix = bigram.fit_transform(documents)
    
    print(f"Bigram vocabulary size: {len(bigram.vocabulary)}")
    print(f"Bigram features: {bigram.get_feature_names()}")
    print(f"Bigram matrix shape: {bigram_matrix.shape}")
    print("Sample bigram matrix (first 2 documents, first 3 features):")
    print(bigram_matrix[:2, :3])
    
    # Trigrams
    trigram = NGramVectorizer(n=3, max_features=10, min_df=1)
    trigram_matrix = trigram.fit_transform(documents)
    
    print(f"\nTrigram vocabulary size: {len(trigram.vocabulary)}")
    print(f"Trigram features: {trigram.get_feature_names()}")
    print(f"Trigram matrix shape: {trigram_matrix.shape}")
    
    # Comparison of methods
    print("\n" + "="*50)
    print("Comparison of Feature Extraction Methods")
    print("="*50)
    
    # Create a larger sample for comparison
    large_documents = documents * 20  # Repeat documents
    
    import time
    
    # Time BoW
    start_time = time.time()
    bow_large = BagOfWords(max_features=100)
    bow_large_matrix = bow_large.fit_transform(large_documents)
    bow_time = time.time() - start_time
    
    # Time TF-IDF
    start_time = time.time()
    tfidf_large = TFIDFVectorizer(max_features=100)
    tfidf_large_matrix = tfidf_large.fit_transform(large_documents)
    tfidf_time = time.time() - start_time
    
    # Time N-grams
    start_time = time.time()
    ngram_large = NGramVectorizer(n=2, max_features=100)
    ngram_large_matrix = ngram_large.fit_transform(large_documents)
    ngram_time = time.time() - start_time
    
    print(f"Processing {len(large_documents)} documents:")
    print(f"BoW time: {bow_time:.4f} seconds")
    print(f"TF-IDF time: {tfidf_time:.4f} seconds")
    print(f"N-gram time: {ngram_time:.4f} seconds")
    
    print(f"\nMatrix shapes:")
    print(f"BoW: {bow_large_matrix.shape}")
    print(f"TF-IDF: {tfidf_large_matrix.shape}")
    print(f"N-gram: {ngram_large_matrix.shape}")
    
    # Sparsity analysis
    print("\n" + "="*50)
    print("Sparsity Analysis")
    print("="*50)
    
    def calculate_sparsity(matrix):
        """Calculate sparsity of a matrix."""
        total_elements = matrix.size
        zero_elements = np.count_nonzero(matrix == 0)
        sparsity = zero_elements / total_elements
        return sparsity
    
    bow_sparsity = calculate_sparsity(bow_large_matrix)
    tfidf_sparsity = calculate_sparsity(tfidf_large_matrix)
    ngram_sparsity = calculate_sparsity(ngram_large_matrix)
    
    print(f"BoW sparsity: {bow_sparsity:.2%}")
    print(f"TF-IDF sparsity: {tfidf_sparsity:.2%}")
    print(f"N-gram sparsity: {ngram_sparsity:.2%}")
    
    # Feature importance demonstration
    print("\n" + "="*50)
    print("Feature Importance Analysis")
    print("="*50)
    
    # Show top features by TF-IDF scores
    feature_scores = np.mean(tfidf_large_matrix, axis=0)
    top_features_idx = np.argsort(feature_scores)[::-1][:10]
    top_features = [tfidf_large.get_feature_names()[i] for i in top_features_idx]
    top_scores = [feature_scores[i] for i in top_features_idx]
    
    print("Top 10 features by average TF-IDF score:")
    for feature, score in zip(top_features, top_scores):
        print(f"  {feature}: {score:.4f}")
    
    # Advanced feature extraction concepts
    print("\n" + "="*50)
    print("Advanced Feature Extraction Concepts")
    print("="*50)
    print("1. Sublinear TF Scaling:")
    print("   - Apply log transformation to term frequencies")
    print("   - Helps reduce the impact of very frequent terms")
    
    print("\n2. Document Length Normalization:")
    print("   - Normalize by document length to account for varying document sizes")
    print("   - Important for fair comparison between documents")
    
    print("\n3. Feature Selection:")
    print("   - Chi-square test for feature relevance")
    print("   - Mutual information for feature importance")
    print("   - Variance thresholds for removing low-variance features")
    
    print("\n4. Advanced N-grams:")
    print("   - Skip-grams: Allow gaps between words")
    print("   - Character n-grams: For morphological analysis")
    print("   - Mixed n-grams: Combine word and character level")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Feature Extraction")
    print("="*50)
    print("1. Choose Appropriate Methods:")
    print("   - BoW for simple classification tasks")
    print("   - TF-IDF for information retrieval and ranking")
    print("   - N-grams for capturing local context")
    
    print("\n2. Parameter Tuning:")
    print("   - Adjust max_features based on computational resources")
    print("   - Set min_df to remove rare terms")
    print("   - Set max_df to remove too common terms")
    
    print("\n3. Preprocessing Integration:")
    print("   - Apply text preprocessing before feature extraction")
    print("   - Ensure consistent preprocessing across datasets")
    
    print("\n4. Evaluation:")
    print("   - Compare different feature extraction methods")
    print("   - Validate with cross-validation")
    print("   - Monitor for overfitting with high-dimensional features")
    
    print("\n5. Scalability:")
    print("   - Use sparse matrices for memory efficiency")
    print("   - Consider dimensionality reduction techniques")
    print("   - Implement incremental learning for large datasets")