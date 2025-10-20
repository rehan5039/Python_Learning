"""
Natural Language Processing Case Study

This case study demonstrates the application of Data Structures and Algorithms in NLP:
- Efficient text processing and tokenization
- TF-IDF computation optimization
- Similarity algorithms for document comparison
- Memory-efficient handling of large text corpora
- Scalable NLP pipeline implementation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import hashlib
import re
import heapq
import time


class EfficientTokenizer:
    """
    Memory-efficient tokenizer using optimized string operations.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.pattern = re.compile(r'\b\w+\b') if remove_punctuation else None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text efficiently using regex.
        
        Time Complexity: O(n) where n is text length
        Space Complexity: O(k) where k is number of tokens
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation and self.pattern:
            tokens = self.pattern.findall(text)
        else:
            tokens = text.split()
        
        return tokens
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize multiple texts efficiently."""
        return [self.tokenize(text) for text in texts]


class InvertedIndex:
    """
    Inverted index for efficient document retrieval using hash tables.
    """
    
    def __init__(self):
        self.index = defaultdict(set)
        self.doc_lengths = {}
        self.vocabulary = set()
    
    def add_document(self, doc_id: int, tokens: List[str]) -> None:
        """
        Add document to inverted index.
        
        Time Complexity: O(k) where k is number of tokens
        Space Complexity: O(k)
        """
        # Update vocabulary
        self.vocabulary.update(tokens)
        
        # Add tokens to index
        for token in tokens:
            self.index[token].add(doc_id)
        
        # Store document length
        self.doc_lengths[doc_id] = len(tokens)
    
    def get_documents_containing(self, term: str) -> Set[int]:
        """Get documents containing specific term."""
        return self.index.get(term, set())
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get term frequency in document (requires additional storage)."""
        # This is a simplified version - in practice, you'd store term frequencies
        return 1 if doc_id in self.index.get(term, set()) else 0
    
    def get_document_frequency(self, term: str) -> int:
        """Get document frequency of term."""
        return len(self.index.get(term, set()))


class TFIDFVectorizer:
    """
    Optimized TF-IDF vectorizer using sparse matrix techniques.
    """
    
    def __init__(self, max_features: int = 10000, min_df: int = 2, max_df: float = 0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary = {}
        self.idf_values = {}
        self.tokenizer = EfficientTokenizer()
    
    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Fit TF-IDF vectorizer to documents.
        
        Time Complexity: O(n * m) where n is documents, m is avg document length
        Space Complexity: O(v) where v is vocabulary size
        """
        n_docs = len(documents)
        
        # Tokenize all documents
        tokenized_docs = self.tokenizer.tokenize_batch(documents)
        
        # Calculate document frequencies
        doc_freq = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # Filter vocabulary based on min_df and max_df
        valid_terms = []
        for term, df in doc_freq.items():
            if df >= self.min_df and df <= self.max_df * n_docs:
                valid_terms.append((term, df))
        
        # Sort by document frequency and limit features
        valid_terms.sort(key=lambda x: x[1], reverse=True)
        valid_terms = valid_terms[:self.max_features]
        
        # Create vocabulary
        self.vocabulary = {term: i for i, (term, _) in enumerate(valid_terms)}
        
        # Calculate IDF values
        self.idf_values = {}
        for term, df in valid_terms:
            self.idf_values[term] = np.log(n_docs / df)
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.
        
        Time Complexity: O(n * m) where n is documents, m is avg document length
        Space Complexity: O(n * v) where v is vocabulary size
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary)
        
        # Initialize TF-IDF matrix
        tfidf_matrix = np.zeros((n_docs, n_features))
        
        # Process each document
        for doc_idx, document in enumerate(documents):
            tokens = self.tokenizer.tokenize(document)
            doc_length = len(tokens)
            
            if doc_length == 0:
                continue
            
            # Calculate term frequencies
            term_counts = Counter(tokens)
            
            # Calculate TF-IDF for each term
            for term, count in term_counts.items():
                if term in self.vocabulary:
                    term_idx = self.vocabulary[term]
                    tf = count / doc_length
                    idf = self.idf_values[term]
                    tfidf_matrix[doc_idx, term_idx] = tf * idf
        
        return tfidf_matrix
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform documents."""
        return self.fit(documents).transform(documents)


class DocumentSimilarity:
    """
    Efficient document similarity computation using cosine similarity.
    """
    
    @staticmethod
    def cosine_similarity_sparse(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between sparse vectors.
        
        Time Complexity: O(k) where k is non-zero elements
        Space Complexity: O(1)
        """
        # For sparse vectors, we can optimize by only considering non-zero elements
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def find_similar_documents(tfidf_matrix: np.ndarray, 
                             query_idx: int, 
                             top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar documents to query document.
        
        Time Complexity: O(n * m) where n is documents, m is features
        Space Complexity: O(n)
        """
        query_vector = tfidf_matrix[query_idx]
        similarities = []
        
        for doc_idx in range(len(tfidf_matrix)):
            if doc_idx != query_idx:
                similarity = DocumentSimilarity.cosine_similarity_sparse(
                    query_vector, tfidf_matrix[doc_idx])
                similarities.append((doc_idx, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class TextSummarizer:
    """
    Text summarization using extractive methods with efficient algorithms.
    """
    
    def __init__(self):
        self.tokenizer = EfficientTokenizer()
    
    def summarize_tfidf(self, sentences: List[str], num_sentences: int = 3) -> List[str]:
        """
        Summarize text using TF-IDF scores.
        
        Time Complexity: O(n * m + n^2) where n is sentences, m is avg sentence length
        Space Complexity: O(n * v) where v is vocabulary size
        """
        if len(sentences) <= num_sentences:
            return sentences
        
        # Vectorize sentences
        vectorizer = TFIDFVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores (mean of TF-IDF scores)
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = np.mean(tfidf_matrix[i])
            sentence_scores.append((i, score))
        
        # Sort by scores and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sentence_scores[:num_sentences]]
        top_indices.sort()  # Maintain original order
        
        return [sentences[i] for i in top_indices]


class NLPProcessingPipeline:
    """
    Complete NLP processing pipeline with optimized components.
    """
    
    def __init__(self):
        self.tokenizer = EfficientTokenizer()
        self.vectorizer = TFIDFVectorizer()
        self.inverted_index = InvertedIndex()
    
    def process_corpus(self, documents: List[str]) -> Dict:
        """
        Process entire corpus through NLP pipeline.
        
        Time Complexity: O(n * m) where n is documents, m is avg document length
        Space Complexity: O(n * v) where v is vocabulary size
        """
        # Tokenize documents
        tokenized_docs = self.tokenizer.tokenize_batch(documents)
        
        # Build inverted index
        for doc_id, tokens in enumerate(tokenized_docs):
            self.inverted_index.add_document(doc_id, tokens)
        
        # Vectorize documents
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        return {
            'tfidf_matrix': tfidf_matrix,
            'vocabulary_size': len(self.vectorizer.vocabulary),
            'documents_processed': len(documents)
        }


def generate_sample_documents(n_docs: int = 1000) -> List[str]:
    """
    Generate sample documents for NLP demonstration.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    np.random.seed(42)
    sample_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning algorithms are powerful tools for data analysis",
        "Natural language processing enables computers to understand human language",
        "Data science combines statistics programming and domain expertise",
        "Deep learning neural networks can solve complex problems",
        "Python is a popular programming language for data science",
        "Statistical analysis helps extract insights from data",
        "Artificial intelligence is transforming many industries",
        "Big data technologies handle large volumes of information",
        "Computer vision systems can recognize objects in images"
    ]
    
    documents = []
    for i in range(n_docs):
        # Generate document with 5-15 random sentences
        n_sentences = np.random.randint(5, 16)
        document_sentences = np.random.choice(sample_sentences, n_sentences)
        document = ". ".join(document_sentences) + "."
        documents.append(document)
    
    return documents


def performance_evaluation():
    """Evaluate performance of NLP algorithms."""
    print("=== NLP Performance Evaluation ===\n")
    
    # Generate sample documents
    print("1. Generating Sample Documents:")
    documents = generate_sample_documents(n_docs=5000)
    print(f"   Generated {len(documents)} documents")
    avg_length = np.mean([len(doc) for doc in documents])
    print(f"   Average document length: {avg_length:.1f} characters")
    
    # Test EfficientTokenizer
    print("\n2. EfficientTokenizer Performance:")
    tokenizer = EfficientTokenizer()
    start_time = time.time()
    tokenized_docs = tokenizer.tokenize_batch(documents[:1000])
    tokenizer_time = time.time() - start_time
    total_tokens = sum(len(tokens) for tokens in tokenized_docs)
    print(f"   Tokenization time: {tokenizer_time:.4f} seconds")
    print(f"   Total tokens processed: {total_tokens}")
    print(f"   Average tokens per document: {total_tokens / len(tokenized_docs):.1f}")
    
    # Test TFIDFVectorizer
    print("\n3. TFIDFVectorizer Performance:")
    vectorizer = TFIDFVectorizer(max_features=5000)
    start_time = time.time()
    tfidf_matrix = vectorizer.fit_transform(documents[:1000])
    vectorizer_time = time.time() - start_time
    print(f"   Vectorization time: {vectorizer_time:.4f} seconds")
    print(f"   TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"   Sparsity: {1 - np.count_nonzero(tfidf_matrix) / tfidf_matrix.size:.4f}")
    
    # Test DocumentSimilarity
    print("\n4. DocumentSimilarity Performance:")
    start_time = time.time()
    similar_docs = DocumentSimilarity.find_similar_documents(tfidf_matrix, query_idx=0, top_k=10)
    similarity_time = time.time() - start_time
    print(f"   Similarity computation time: {similarity_time:.6f} seconds")
    print(f"   Most similar document index: {similar_docs[0][0]}, similarity: {similar_docs[0][1]:.4f}")
    
    # Test TextSummarizer
    print("\n5. TextSummarizer Performance:")
    summarizer = TextSummarizer()
    sample_doc = documents[0]
    sentences = sample_doc.split('. ')
    start_time = time.time()
    summary = summarizer.summarize_tfidf(sentences, num_sentences=3)
    summary_time = time.time() - start_time
    print(f"   Summarization time: {summary_time:.6f} seconds")
    print(f"   Original sentences: {len(sentences)}")
    print(f"   Summary sentences: {len(summary)}")
    
    # Test NLPProcessingPipeline
    print("\n6. NLPProcessingPipeline Performance:")
    pipeline = NLPProcessingPipeline()
    start_time = time.time()
    results = pipeline.process_corpus(documents[:500])
    pipeline_time = time.time() - start_time
    print(f"   Pipeline processing time: {pipeline_time:.4f} seconds")
    print(f"   Documents processed: {results['documents_processed']}")
    print(f"   Vocabulary size: {results['vocabulary_size']}")
    print(f"   TF-IDF matrix shape: {results['tfidf_matrix'].shape}")


def demo():
    """Demonstrate NLP case study."""
    print("=== Natural Language Processing Case Study ===\n")
    
    # Generate sample documents
    documents = generate_sample_documents(n_docs=100)
    print("Sample documents generated:")
    print(f"  Number of documents: {len(documents)}")
    print(f"  Sample document 0: {documents[0][:100]}...")
    
    # Demonstrate EfficientTokenizer
    print("\n1. EfficientTokenizer:")
    tokenizer = EfficientTokenizer()
    tokens = tokenizer.tokenize(documents[0])
    print(f"  Tokenized first document: {len(tokens)} tokens")
    print(f"  First 10 tokens: {tokens[:10]}")
    
    # Demonstrate TFIDFVectorizer
    print("\n2. TFIDFVectorizer:")
    vectorizer = TFIDFVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(documents[:20])
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary)}")
    print(f"  Sample TF-IDF values (first document): {tfidf_matrix[0][:10]}")
    
    # Demonstrate DocumentSimilarity
    print("\n3. DocumentSimilarity:")
    similar_docs = DocumentSimilarity.find_similar_documents(tfidf_matrix, query_idx=0, top_k=3)
    print(f"  Most similar documents to document 0:")
    for doc_idx, similarity in similar_docs:
        print(f"    Document {doc_idx}: similarity = {similarity:.4f}")
    
    # Demonstrate TextSummarizer
    print("\n4. TextSummarizer:")
    summarizer = TextSummarizer()
    sample_sentences = documents[0].split('. ')[:10]  # First 10 sentences
    summary = summarizer.summarize_tfidf(sample_sentences, num_sentences=3)
    print(f"  Original sentences: {len(sample_sentences)}")
    print(f"  Summary sentences: {len(summary)}")
    print("  Summary:")
    for i, sentence in enumerate(summary):
        print(f"    {i+1}. {sentence}")
    
    # Demonstrate NLPProcessingPipeline
    print("\n5. NLPProcessingPipeline:")
    pipeline = NLPProcessingPipeline()
    results = pipeline.process_corpus(documents[:10])
    print(f"  Processed {results['documents_processed']} documents")
    print(f"  Vocabulary size: {results['vocabulary_size']}")
    
    # Performance evaluation
    print("\n" + "="*60)
    performance_evaluation()


if __name__ == "__main__":
    demo()