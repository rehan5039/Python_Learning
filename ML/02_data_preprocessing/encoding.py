"""
Data Preprocessing - Encoding Categorical Variables
============================================

This module covers techniques for encoding categorical variables for machine learning.
Proper encoding is essential since most ML algorithms require numerical input.

Topics Covered:
- Label encoding
- One-hot encoding
- Ordinal encoding
- Binary encoding
- Target encoding
- Handling high cardinality categorical variables
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def label_encoding(df, column):
    """
    Apply label encoding to a categorical column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to encode
    
    Returns:
        tuple: (encoded_series, encoder)
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[column])
    return pd.Series(encoded, name=f'{column}_encoded'), encoder

def one_hot_encoding(df, columns, drop_first=False):
    """
    Apply one-hot encoding to categorical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list): List of column names to encode
        drop_first (bool): Whether to drop first category to avoid multicollinearity
    
    Returns:
        tuple: (encoded_dataframe, encoder)
    """
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse=False)
    encoded = encoder.fit_transform(df[columns])
    
    # Create column names
    feature_names = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded, columns=feature_names)
    
    return encoded_df, encoder

def ordinal_encoding(df, column, categories=None):
    """
    Apply ordinal encoding to a categorical column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to encode
        categories (list): Ordered categories (if None, inferred from data)
    
    Returns:
        tuple: (encoded_series, encoder)
    """
    if categories is None:
        # Infer order from frequency (most frequent first)
        categories = df[column].value_counts().index.tolist()
    
    # Create mapping
    category_mapping = {cat: i for i, cat in enumerate(categories)}
    encoded = df[column].map(category_mapping)
    
    return pd.Series(encoded, name=f'{column}_ordinal'), category_mapping

def binary_encoding(df, column):
    """
    Apply binary encoding to a categorical column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to encode
    
    Returns:
        pd.DataFrame: Binary encoded columns
    """
    encoder = ce.BinaryEncoder(cols=[column])
    encoded = encoder.fit_transform(df)
    return encoded

def target_encoding(df, column, target, smoothing=1.0):
    """
    Apply target encoding to a categorical column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to encode
        target (str): Target column name
        smoothing (float): Smoothing parameter
    
    Returns:
        tuple: (encoded_series, mapping)
    """
    # Calculate global mean
    global_mean = df[target].mean()
    
    # Calculate category means and counts
    category_stats = df.groupby(column)[target].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothed_means = (category_stats['mean'] * category_stats['count'] + 
                     global_mean * smoothing) / (category_stats['count'] + smoothing)
    
    # Create mapping
    encoding_map = smoothed_means.to_dict()
    
    # Apply encoding
    encoded = df[column].map(encoding_map).fillna(global_mean)
    
    return pd.Series(encoded, name=f'{column}_target'), encoding_map

def frequency_encoding(df, column):
    """
    Apply frequency encoding to a categorical column
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to encode
    
    Returns:
        tuple: (encoded_series, frequency_map)
    """
    # Calculate frequency of each category
    frequency_map = df[column].value_counts().to_dict()
    
    # Apply encoding
    encoded = df[column].map(frequency_map)
    
    return pd.Series(encoded, name=f'{column}_frequency'), frequency_map

def handle_high_cardinality(df, column, method='frequency', threshold=10):
    """
    Handle high cardinality categorical variables
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        method (str): Method to handle high cardinality ('frequency', 'target', 'grouping')
        threshold (int): Cardinality threshold
    
    Returns:
        pd.DataFrame: DataFrame with handled high cardinality
    """
    df_copy = df.copy()
    
    # Check cardinality
    unique_count = df_copy[column].nunique()
    
    if unique_count <= threshold:
        return df_copy  # Low cardinality, no special handling needed
    
    if method == 'frequency':
        # Replace rare categories with 'Other'
        frequency = df_copy[column].value_counts()
        rare_categories = frequency[frequency < threshold].index
        df_copy[column] = df_copy[column].replace(rare_categories, 'Other')
    
    elif method == 'target':
        # Group by target mean similarity (simplified approach)
        # In practice, you might use clustering or other similarity measures
        pass
    
    elif method == 'grouping':
        # Manual grouping based on domain knowledge
        # This would require specific grouping rules
        pass
    
    return df_copy

def compare_encoding_methods(df, categorical_columns, target_column=None):
    """
    Compare different encoding methods on categorical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_columns (list): List of categorical column names
        target_column (str): Target column for target encoding
    
    Returns:
        dict: Dictionary containing encoded data from different methods
    """
    results = {}
    
    for column in categorical_columns:
        print(f"Encoding column: {column}")
        
        # Label Encoding
        try:
            encoded, _ = label_encoding(df, column)
            results[f'{column}_label'] = encoded
        except Exception as e:
            print(f"Label encoding failed for {column}: {e}")
        
        # Ordinal Encoding (based on frequency)
        try:
            encoded, _ = ordinal_encoding(df, column)
            results[f'{column}_ordinal'] = encoded
        except Exception as e:
            print(f"Ordinal encoding failed for {column}: {e}")
        
        # Frequency Encoding
        try:
            encoded, _ = frequency_encoding(df, column)
            results[f'{column}_frequency'] = encoded
        except Exception as e:
            print(f"Frequency encoding failed for {column}: {e}")
        
        # Target Encoding (if target column provided)
        if target_column and target_column in df.columns:
            try:
                encoded, _ = target_encoding(df, column, target_column)
                results[f'{column}_target'] = encoded
            except Exception as e:
                print(f"Target encoding failed for {column}: {e}")
    
    return results

def encoding_demo():
    """
    Demonstrate categorical encoding techniques
    """
    print("=== Categorical Encoding Demo ===")
    
    # Create sample dataset with categorical variables
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance', 'IT'], n_samples),
        'experience_level': np.random.choice(['Junior', 'Mid', 'Senior', 'Lead'], n_samples),
        'salary': np.random.normal(70000, 20000, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create binary target based on salary
    df['high_salary'] = (df['salary'] > df['salary'].median()).astype(int)
    
    print("1. Original Data Info:")
    print(f"   Shape: {df.shape}")
    print("\n   Categorical columns:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"   {col}: {df[col].nunique()} unique values")
        print(f"      Values: {df[col].value_counts().head()}")
    
    # Label Encoding
    print("\n2. Label Encoding:")
    for col in categorical_columns:
        encoded, encoder = label_encoding(df, col)
        print(f"   {col}: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    
    # One-Hot Encoding
    print("\n3. One-Hot Encoding:")
    encoded_df, encoder = one_hot_encoding(df, categorical_columns.tolist(), drop_first=True)
    print(f"   Encoded shape: {encoded_df.shape}")
    print(f"   First 5 columns: {encoded_df.columns[:5].tolist()}")
    
    # Ordinal Encoding
    print("\n4. Ordinal Encoding (by frequency):")
    for col in categorical_columns:
        encoded, mapping = ordinal_encoding(df, col)
        print(f"   {col}: {mapping}")
    
    # Target Encoding
    print("\n5. Target Encoding:")
    for col in categorical_columns:
        encoded, mapping = target_encoding(df, col, 'high_salary')
        print(f"   {col}: {dict(list(mapping.items())[:3])}...")  # Show first 3 mappings
    
    # Frequency Encoding
    print("\n6. Frequency Encoding:")
    for col in categorical_columns:
        encoded, mapping = frequency_encoding(df, col)
        print(f"   {col}: {dict(list(mapping.items())[:3])}...")  # Show first 3 mappings

def when_to_use_encoding():
    """
    Guidelines for when to use different encoding methods
    """
    print("\n=== When to Use Different Encoding Methods ===")
    
    guidelines = {
        "Label Encoding": {
            "When to use": "For ordinal categorical variables with natural ordering",
            "Benefits": "Simple, preserves ordering information, memory efficient",
            "Drawbacks": "Implies ordering for nominal variables, may mislead algorithms",
            "Best for": "Tree-based algorithms, ordinal variables"
        },
        "One-Hot Encoding": {
            "When to use": "For nominal categorical variables with low cardinality",
            "Benefits": "No ordering assumption, works well with linear models",
            "Drawbacks": "Curse of dimensionality with high cardinality, multicollinearity",
            "Best for": "Linear models, SVM, when categories are truly nominal"
        },
        "Ordinal Encoding": {
            "When to use": "When you can define meaningful order for categories",
            "Benefits": "Preserves ordinal relationships, memory efficient",
            "Drawbacks": "Requires domain knowledge for ordering",
            "Best for": "When ordinal relationships exist in data"
        },
        "Binary Encoding": {
            "When to use": "For high cardinality categorical variables",
            "Benefits": "Reduces dimensionality compared to one-hot, preserves information",
            "Drawbacks": "Less interpretable, may create artificial relationships",
            "Best for": "High cardinality nominal variables"
        },
        "Target Encoding": {
            "When to use": "For high cardinality variables in supervised learning",
            "Benefits": "Captures relationship with target, reduces dimensionality",
            "Drawbacks": "Risk of overfitting, data leakage if not implemented carefully",
            "Best for": "High cardinality variables in supervised learning"
        },
        "Frequency Encoding": {
            "When to use": "When frequency of categories is informative",
            "Benefits": "Simple, captures popularity information",
            "Drawbacks": "May not capture relationship with target",
            "Best for": "When category frequency is meaningful"
        }
    }
    
    for method, info in guidelines.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

def handle_categorical_pipeline(df, categorical_columns, target_column=None):
    """
    Create a comprehensive pipeline for handling categorical variables
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_columns (list): List of categorical column names
        target_column (str): Target column name for target encoding
    
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    df_processed = df.copy()
    
    for column in categorical_columns:
        print(f"Processing column: {column}")
        
        # Check cardinality
        cardinality = df_processed[column].nunique()
        print(f"  Cardinality: {cardinality}")
        
        if cardinality == 1:
            # Drop constant columns
            df_processed.drop(column, axis=1, inplace=True)
            print("  Dropped constant column")
        elif cardinality == 2:
            # Binary column - use label encoding
            encoded, _ = label_encoding(df_processed, column)
            df_processed.drop(column, axis=1, inplace=True)
            df_processed[encoded.name] = encoded
            print("  Applied label encoding for binary column")
        elif cardinality <= 10:
            # Low cardinality - use one-hot encoding
            encoded_df, _ = one_hot_encoding(df_processed, [column], drop_first=True)
            df_processed.drop(column, axis=1, inplace=True)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            print("  Applied one-hot encoding")
        else:
            # High cardinality - use target or frequency encoding
            if target_column and target_column in df_processed.columns:
                try:
                    encoded, _ = target_encoding(df_processed, column, target_column)
                    df_processed.drop(column, axis=1, inplace=True)
                    df_processed[encoded.name] = encoded
                    print("  Applied target encoding")
                except:
                    encoded, _ = frequency_encoding(df_processed, column)
                    df_processed.drop(column, axis=1, inplace=True)
                    df_processed[encoded.name] = encoded
                    print("  Applied frequency encoding (target encoding failed)")
            else:
                encoded, _ = frequency_encoding(df_processed, column)
                df_processed.drop(column, axis=1, inplace=True)
                df_processed[encoded.name] = encoded
                print("  Applied frequency encoding")
    
    return df_processed

def encoding_impact_on_algorithms():
    """
    Demonstrate how encoding impacts different ML algorithms
    """
    print("\n=== Impact of Encoding on ML Algorithms ===")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'category_A': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'category_B': np.random.choice(['Type1', 'Type2', 'Type3', 'Type4', 'Type5'], n_samples),
        'numerical_feature': np.random.normal(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    df['target'] = ((df['numerical_feature'] > 0) & 
                   (df['category_A'].isin(['Medium', 'High']))).astype(int)
    
    # Split data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Different encoding approaches
    encodings = {
        'Label Encoding': {},
        'One-Hot Encoding': {},
        'Frequency Encoding': {}
    }
    
    # Test with different algorithms
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    print("Algorithm Performance with Different Encoding Methods:")
    print("-" * 60)
    
    for algo_name, algorithm in algorithms.items():
        print(f"\n{algo_name}:")
        
        # Label Encoding
        X_train_label = X_train.copy()
        X_test_label = X_test.copy()
        
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train_label[col] = le.fit_transform(X_train_label[col])
            X_test_label[col] = le.transform(X_test_label[col])
        
        algorithm.fit(X_train_label, y_train)
        y_pred = algorithm.predict(X_test_label)
        accuracy_label = accuracy_score(y_test, y_pred)
        
        # One-Hot Encoding
        X_train_ohe = pd.get_dummies(X_train, drop_first=True)
        X_test_ohe = pd.get_dummies(X_test, drop_first=True)
        
        # Ensure same columns
        missing_cols = set(X_train_ohe.columns) - set(X_test_ohe.columns)
        for col in missing_cols:
            X_test_ohe[col] = 0
        X_test_ohe = X_test_ohe[X_train_ohe.columns]
        
        algorithm.fit(X_train_ohe, y_train)
        y_pred = algorithm.predict(X_test_ohe)
        accuracy_ohe = accuracy_score(y_test, y_pred)
        
        print(f"   Label Encoding: {accuracy_label:.4f}")
        print(f"   One-Hot Encoding: {accuracy_ohe:.4f}")

# Example usage and testing
if __name__ == "__main__":
    # Encoding demo
    encoding_demo()
    print("\n" + "="*50 + "\n")
    
    # When to use different encoding methods
    when_to_use_encoding()
    print("\n" + "="*50 + "\n")
    
    # Impact on algorithms
    encoding_impact_on_algorithms()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Various categorical encoding techniques and their implementations")
    print("2. Comparison of different encoding methods")
    print("3. When to use each encoding method")
    print("4. Impact of encoding on machine learning algorithms")
    print("5. Handling high cardinality categorical variables")
    print("6. Creating comprehensive encoding pipelines")
    print("\nKey takeaways:")
    print("- Choice of encoding method depends on variable type and algorithm")
    print("- High cardinality variables require special handling")
    print("- Target encoding can be powerful but requires careful implementation")
    print("- Always validate that encoding improves model performance")
    print("- Create reproducible encoding pipelines for consistent results")