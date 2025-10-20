"""
Feature Engineering with DSA Principles

This module demonstrates how to apply Data Structures and Algorithms concepts to feature engineering:
- Efficient feature creation and transformation
- Dimensionality reduction techniques
- Feature selection algorithms
- Text processing optimizations
- Time series feature extraction
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import hashlib
import re


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features efficiently using combinatorial algorithms.
    
    Time Complexity: O(n * d^degree) where n samples, d features
    Space Complexity: O(n * C(d+degree, degree))
    """
    n_samples, n_features = X.shape
    
    if degree == 1:
        return X
    
    # Start with original features
    features = [X]
    
    # Generate polynomial features
    for d in range(2, degree + 1):
        # For each sample, compute polynomial combinations
        poly_features = np.ones((n_samples, 1))
        
        # Simple approach: multiply all feature combinations
        # In practice, you'd use more sophisticated combinatorial generation
        for i in range(n_features):
            for j in range(i, n_features):
                if d == 2:
                    new_feature = (X[:, i] * X[:, j]).reshape(-1, 1)
                    poly_features = np.hstack([poly_features, new_feature])
        
        # Remove the initial column of ones (except for degree 1)
        if d > 1:
            poly_features = poly_features[:, 1:]
        
        features.append(poly_features)
    
    return np.hstack(features)


def binning_optimization(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Optimize binning using efficient sorting and partitioning algorithms.
    
    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n)
    """
    # Use quantile-based binning for better distribution
    n_samples = X.shape[0]
    
    # Sort the data
    sorted_indices = np.argsort(X, axis=0)
    sorted_data = np.take_along_axis(X, sorted_indices, axis=0)
    
    # Calculate bin boundaries
    bin_size = n_samples // n_bins
    bin_boundaries = []
    
    for i in range(n_bins + 1):
        if i == n_bins:
            bin_boundaries.append(X.max())
        else:
            idx = min(i * bin_size, n_samples - 1)
            bin_boundaries.append(sorted_data[idx])
    
    # Assign bins
    bins = np.digitize(X, bin_boundaries[:-1])
    
    return bins


def categorical_encoding_optimization(categories: List[str], method: str = 'hash') -> np.ndarray:
    """
    Optimize categorical encoding using appropriate data structures.
    
    Time Complexity: O(n) for hash encoding, O(n log k) for target encoding
    Space Complexity: O(k) where k is number of unique categories
    """
    n_samples = len(categories)
    unique_categories = list(set(categories))
    
    if method == 'label':
        # Label encoding
        category_to_label = {cat: i for i, cat in enumerate(unique_categories)}
        encoded = np.array([category_to_label[cat] for cat in categories])
        
    elif method == 'onehot':
        # One-hot encoding
        category_to_index = {cat: i for i, cat in enumerate(unique_categories)}
        encoded = np.zeros((n_samples, len(unique_categories)))
        for i, cat in enumerate(categories):
            encoded[i, category_to_index[cat]] = 1
            
    elif method == 'hash':
        # Hash encoding (dimensionality reduction)
        encoded = np.zeros((n_samples, 10))  # Fixed dimension
        for i, cat in enumerate(categories):
            # Simple hash function
            hash_val = int(hashlib.md5(cat.encode()).hexdigest(), 16)
            encoded[i, hash_val % 10] = 1
            
    elif method == 'target':
        # Target encoding (requires target variable)
        # This is a simplified version
        category_counts = Counter(categories)
        encoded = np.array([category_counts[cat] for cat in categories])
        
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    
    return encoded


def text_feature_extraction_optimization(texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Optimize text feature extraction using efficient string algorithms.
    
    Time Complexity: O(n * m) where n texts, m average text length
    Space Complexity: O(v) where v is vocabulary size
    """
    # Tokenization with regex (more efficient than split)
    token_pattern = re.compile(r'\b\w+\b')
    
    # Build vocabulary
    vocab = set()
    tokenized_texts = []
    
    for text in texts:
        tokens = token_pattern.findall(text.lower())
        tokenized_texts.append(tokens)
        vocab.update(tokens)
    
    # Convert to list for indexing
    vocab_list = sorted(list(vocab))
    vocab_to_index = {word: i for i, word in enumerate(vocab_list)}
    
    # TF-IDF calculation
    n_texts = len(texts)
    n_vocab = len(vocab_list)
    
    # Term frequency
    tf_matrix = np.zeros((n_texts, n_vocab))
    for i, tokens in enumerate(tokenized_texts):
        token_counts = Counter(tokens)
        for token, count in token_counts.items():
            if token in vocab_to_index:
                tf_matrix[i, vocab_to_index[token]] = count / len(tokens)
    
    # Document frequency
    df_vector = np.zeros(n_vocab)
    for tokens in tokenized_texts:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in vocab_to_index:
                df_vector[vocab_to_index[token]] += 1
    
    # IDF calculation
    idf_vector = np.log(n_texts / (df_vector + 1))
    
    # TF-IDF matrix
    tfidf_matrix = tf_matrix * idf_vector
    
    return {
        'tfidf': tfidf_matrix,
        'vocabulary': vocab_list,
        'idf': idf_vector
    }


def datetime_feature_extraction_optimization(dates: List[str]) -> np.ndarray:
    """
    Extract datetime features using efficient parsing algorithms.
    
    Time Complexity: O(n) where n dates
    Space Complexity: O(n)
    """
    # Convert to datetime objects efficiently
    datetime_objects = pd.to_datetime(dates)
    
    # Extract features
    features = np.column_stack([
        datetime_objects.year,
        datetime_objects.month,
        datetime_objects.day,
        datetime_objects.dayofweek,  # Monday=0, Sunday=6
        datetime_objects.dayofyear,
        datetime_objects.hour,
        datetime_objects.minute
    ])
    
    return features


def feature_selection_optimization(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Optimize feature selection using correlation and statistical tests.
    
    Time Complexity: O(n * m^2) where n samples, m features
    Space Complexity: O(m)
    """
    n_samples, n_features = X.shape
    
    # Calculate correlation with target
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
    correlations = np.abs(correlations)  # Take absolute values
    
    # Select top k features
    selected_indices = np.argsort(correlations)[-k:]
    
    return selected_indices


def automated_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an automated feature engineering pipeline.
    
    Time Complexity: O(n * m) where n samples, m features
    Space Complexity: O(n * m')
    """
    engineered_df = df.copy()
    
    # Numerical feature engineering
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        # Add polynomial features
        engineered_df[f'{col}_squared'] = df[col] ** 2
        engineered_df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        engineered_df[f'{col}_log'] = np.log(np.abs(df[col]) + 1)
        
        # Binning
        engineered_df[f'{col}_binned'] = binning_optimization(df[col].values.reshape(-1, 1)).flatten()
    
    # Categorical feature engineering
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        # Label encoding
        encoded = categorical_encoding_optimization(df[col].tolist(), method='label')
        engineered_df[f'{col}_encoded'] = encoded
    
    # DateTime feature engineering
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        datetime_features = datetime_feature_extraction_optimization(df[col].astype(str).tolist())
        # Add basic datetime features
        engineered_df[f'{col}_year'] = pd.to_datetime(df[col]).year
        engineered_df[f'{col}_month'] = pd.to_datetime(df[col]).month
        engineered_df[f'{col}_day'] = pd.to_datetime(df[col]).day
    
    return engineered_df


def performance_comparison():
    """Compare performance of different feature engineering techniques."""
    print("=== Feature Engineering Performance Comparison ===\n")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    
    # Numerical data
    numerical_data = np.random.randn(n_samples, 5)
    
    # Categorical data
    categories = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    
    # Text data
    texts = [f"sample text {i} with some words" for i in range(1000)]
    
    # DateTime data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D').strftime('%Y-%m-%d').tolist()
    
    # Test polynomial features
    print("1. Polynomial Features:")
    import time
    start_time = time.time()
    poly_features = polynomial_features(numerical_data, degree=2)
    poly_time = time.time() - start_time
    print(f"   Polynomial features time: {poly_time:.6f} seconds")
    print(f"   Original shape: {numerical_data.shape}")
    print(f"   Polynomial shape: {poly_features.shape}")
    
    # Test binning
    print("\n2. Binning Optimization:")
    start_time = time.time()
    binned = binning_optimization(numerical_data[:, 0], n_bins=10)
    binning_time = time.time() - start_time
    print(f"   Binning time: {binning_time:.6f} seconds")
    print(f"   Unique bins: {len(np.unique(binned))}")
    
    # Test categorical encoding
    print("\n3. Categorical Encoding:")
    start_time = time.time()
    encoded = categorical_encoding_optimization(categories[:1000], method='hash')
    encoding_time = time.time() - start_time
    print(f"   Encoding time: {encoding_time:.6f} seconds")
    print(f"   Encoded shape: {encoded.shape}")
    
    # Test text feature extraction
    print("\n4. Text Feature Extraction:")
    start_time = time.time()
    text_features = text_feature_extraction_optimization(texts)
    text_time = time.time() - start_time
    print(f"   Text extraction time: {text_time:.6f} seconds")
    print(f"   TF-IDF shape: {text_features['tfidf'].shape}")
    print(f"   Vocabulary size: {len(text_features['vocabulary'])}")
    
    # Test datetime feature extraction
    print("\n5. DateTime Feature Extraction:")
    start_time = time.time()
    datetime_features = datetime_feature_extraction_optimization(dates)
    datetime_time = time.time() - start_time
    print(f"   DateTime extraction time: {datetime_time:.6f} seconds")
    print(f"   DateTime features shape: {datetime_features.shape}")
    
    # Test feature selection
    print("\n6. Feature Selection:")
    target = numerical_data[:, 0] + np.random.randn(n_samples) * 0.1
    start_time = time.time()
    selected_features = feature_selection_optimization(numerical_data, target, k=3)
    selection_time = time.time() - start_time
    print(f"   Feature selection time: {selection_time:.6f} seconds")
    print(f"   Selected features: {selected_features}")


def demo():
    """Demonstrate feature engineering techniques."""
    print("=== Feature Engineering with DSA ===\n")
    
    # Create sample DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'numerical_1': np.random.randn(1000),
        'numerical_2': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'date': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'text': [f"sample text {i}" for i in range(1000)]
    })
    
    print("Original DataFrame:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    # Test polynomial features
    print("\n1. Polynomial Features:")
    numerical_data = df[['numerical_1', 'numerical_2']].values
    poly_features = polynomial_features(numerical_data, degree=2)
    print(f"  Original shape: {numerical_data.shape}")
    print(f"  Polynomial shape: {poly_features.shape}")
    
    # Test binning
    print("\n2. Binning:")
    binned = binning_optimization(df['numerical_1'].values.reshape(-1, 1))
    print(f"  Binned values (first 10): {binned.flatten()[:10]}")
    print(f"  Unique bins: {len(np.unique(binned))}")
    
    # Test categorical encoding
    print("\n3. Categorical Encoding:")
    encoded = categorical_encoding_optimization(df['category'].tolist(), method='hash')
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Sample encoded values: {encoded[:5]}")
    
    # Test datetime feature extraction
    print("\n4. DateTime Feature Extraction:")
    datetime_features = datetime_feature_extraction_optimization(df['date'].astype(str).tolist())
    print(f"  DateTime features shape: {datetime_features.shape}")
    print(f"  Sample features: {datetime_features[:3]}")
    
    # Test automated pipeline
    print("\n5. Automated Feature Engineering Pipeline:")
    engineered_df = automated_feature_engineering_pipeline(df.head(100))
    print(f"  Engineered DataFrame shape: {engineered_df.shape}")
    print(f"  New columns: {set(engineered_df.columns) - set(df.columns)}")
    
    # Performance comparison
    print("\n" + "="*60)
    performance_comparison()


if __name__ == "__main__":
    demo()