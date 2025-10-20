"""
Data Preprocessing - Feature Engineering
==================================

This module covers techniques for creating new features from existing data to
improve machine learning model performance. Feature engineering is often the
difference between a good model and a great model.

Topics Covered:
- Polynomial features
- Interaction features
- Binning and discretization
- Date and time feature extraction
- Text feature extraction
- Domain-specific feature creation
- Feature selection techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_polynomial_features(X, degree=2, interaction_only=False):
    """
    Create polynomial and interaction features
    
    Args:
        X (array-like): Input features
        degree (int): Degree of polynomial features
        interaction_only (bool): If True, only interaction features are produced
    
    Returns:
        tuple: (polynomial_features, feature_names)
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out()
    
    return X_poly, feature_names

def create_interaction_features(df, feature_pairs):
    """
    Create interaction features from pairs of numerical features
    
    Args:
        df (pd.DataFrame): Input DataFrame
        feature_pairs (list): List of tuples containing feature pairs
    
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    df_copy = df.copy()
    
    for feature1, feature2 in feature_pairs:
        if feature1 in df_copy.columns and feature2 in df_copy.columns:
            interaction_name = f"{feature1}_x_{feature2}"
            df_copy[interaction_name] = df_copy[feature1] * df_copy[feature2]
    
    return df_copy

def binning_features(df, column, bins, labels=None):
    """
    Create binned features from continuous variables
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to bin
        bins (int or list): Number of bins or bin edges
        labels (list): Labels for bins
    
    Returns:
        pd.DataFrame: DataFrame with binned features
    """
    df_copy = df.copy()
    
    # Create binned feature
    binned_name = f"{column}_binned"
    df_copy[binned_name] = pd.cut(df_copy[column], bins=bins, labels=labels)
    
    # Create one-hot encoded bins
    if labels is None:
        df_copy = pd.get_dummies(df_copy, columns=[binned_name], prefix=column)
    else:
        # Convert to one-hot with custom labels
        df_copy = pd.get_dummies(df_copy, columns=[binned_name], prefix=column)
    
    return df_copy

def extract_date_features(df, date_column):
    """
    Extract features from date/time columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_column (str): Date column name
    
    Returns:
        pd.DataFrame: DataFrame with extracted date features
    """
    df_copy = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Extract date components
    df_copy[f'{date_column}_year'] = df_copy[date_column].dt.year
    df_copy[f'{date_column}_month'] = df_copy[date_column].dt.month
    df_copy[f'{date_column}_day'] = df_copy[date_column].dt.day
    df_copy[f'{date_column}_dayofweek'] = df_copy[date_column].dt.dayofweek
    df_copy[f'{date_column}_dayofyear'] = df_copy[date_column].dt.dayofyear
    df_copy[f'{date_column}_week'] = df_copy[date_column].dt.isocalendar().week
    df_copy[f'{date_column}_quarter'] = df_copy[date_column].dt.quarter
    
    # Cyclical features (using sine and cosine)
    df_copy[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df_copy[f'{date_column}_month'] / 12)
    df_copy[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df_copy[f'{date_column}_month'] / 12)
    df_copy[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df_copy[f'{date_column}_day'] / 31)
    df_copy[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df_copy[f'{date_column}_day'] / 31)
    
    # Additional features
    df_copy[f'{date_column}_is_weekend'] = df_copy[f'{date_column}_dayofweek'].isin([5, 6]).astype(int)
    df_copy[f'{date_column}_is_month_start'] = (df_copy[f'{date_column}_day'] <= 7).astype(int)
    df_copy[f'{date_column}_is_month_end'] = (df_copy[f'{date_column}_day'] >= 25).astype(int)
    
    return df_copy

def extract_text_features(df, text_column, max_features=100):
    """
    Extract basic features from text columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        text_column (str): Text column name
        max_features (int): Maximum number of features to extract
    
    Returns:
        pd.DataFrame: DataFrame with extracted text features
    """
    df_copy = df.copy()
    
    # Basic text statistics
    df_copy[f'{text_column}_length'] = df_copy[text_column].astype(str).str.len()
    df_copy[f'{text_column}_word_count'] = df_copy[text_column].astype(str).str.split().str.len()
    df_copy[f'{text_column}_avg_word_length'] = (
        df_copy[f'{text_column}_length'] / df_copy[f'{text_column}_word_count']
    ).fillna(0)
    
    # Character-based features
    df_copy[f'{text_column}_uppercase_ratio'] = (
        df_copy[text_column].astype(str).str.count(r'[A-Z]') / df_copy[f'{text_column}_length']
    ).fillna(0)
    
    df_copy[f'{text_column}_digit_ratio'] = (
        df_copy[text_column].astype(str).str.count(r'[0-9]') / df_copy[f'{text_column}_length']
    ).fillna(0)
    
    df_copy[f'{text_column}_punctuation_ratio'] = (
        df_copy[text_column].astype(str).str.count(r'[^\w\s]') / df_copy[f'{text_column}_length']
    ).fillna(0)
    
    # Keyword presence (example keywords)
    keywords = ['important', 'urgent', 'asap', 'please', 'thank']
    for keyword in keywords[:5]:  # Limit for demo
        df_copy[f'{text_column}_{keyword}_present'] = (
            df_copy[text_column].astype(str).str.lower().str.contains(keyword, na=False)
        ).astype(int)
    
    return df_copy

def domain_specific_features(df):
    """
    Create domain-specific features (example for e-commerce data)
    
    Args:
        df (pd.DataFrame): Input DataFrame with e-commerce features
    
    Returns:
        pd.DataFrame: DataFrame with domain-specific features
    """
    df_copy = df.copy()
    
    # Example: E-commerce features
    if 'price' in df_copy.columns and 'discount' in df_copy.columns:
        df_copy['discount_percentage'] = (df_copy['discount'] / df_copy['price']) * 100
        df_copy['final_price'] = df_copy['price'] - df_copy['discount']
        df_copy['price_category'] = pd.cut(df_copy['final_price'], 
                                         bins=[0, 50, 100, 200, float('inf')], 
                                         labels=['Budget', 'Mid', 'Premium', 'Luxury'])
    
    if 'rating' in df_copy.columns and 'review_count' in df_copy.columns:
        # Weighted rating (Bayesian average)
        C = df_copy['rating'].mean()
        m = df_copy['review_count'].quantile(0.75)
        df_copy['weighted_rating'] = (
            (df_copy['review_count'] / (df_copy['review_count'] + m)) * df_copy['rating'] +
            (m / (df_copy['review_count'] + m)) * C
        )
    
    if 'views' in df_copy.columns and 'purchases' in df_copy.columns:
        df_copy['conversion_rate'] = (df_copy['purchases'] / df_copy['views']) * 100
    
    return df_copy

def feature_selection_univariate(X, y, k=10, method='f_classif'):
    """
    Select top k features using univariate statistical tests
    
    Args:
        X (array-like): Input features
        y (array-like): Target variable
        k (int): Number of features to select
        method (str): Statistical test method ('f_classif', 'mutual_info_classif')
    
    Returns:
        tuple: (selected_features, selector, feature_scores)
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info_classif':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'f_classif' or 'mutual_info_classif'")
    
    X_selected = selector.fit_transform(X, y)
    feature_scores = selector.scores_
    
    return X_selected, selector, feature_scores

def feature_selection_pca(X, n_components=0.95):
    """
    Select features using Principal Component Analysis
    
    Args:
        X (array-like): Input features
        n_components (int or float): Number of components or variance ratio
    
    Returns:
        tuple: (transformed_features, pca, explained_variance_ratio)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    
    return X_pca, pca, explained_variance

def create_aggregate_features(df, group_column, agg_columns, agg_functions=['mean', 'std', 'count']):
    """
    Create aggregate features for grouped data
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_column (str): Column to group by
        agg_columns (list): Columns to aggregate
        agg_functions (list): Aggregation functions to apply
    
    Returns:
        pd.DataFrame: DataFrame with aggregate features
    """
    df_copy = df.copy()
    
    # Create aggregate features
    agg_dict = {col: agg_functions for col in agg_columns}
    group_stats = df_copy.groupby(group_column).agg(agg_dict)
    
    # Flatten column names
    group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
    group_stats.reset_index(inplace=True)
    
    # Merge back with original data
    df_copy = df_copy.merge(group_stats, on=group_column, how='left')
    
    return df_copy

def feature_engineering_demo():
    """
    Demonstrate feature engineering techniques
    """
    print("=== Feature Engineering Demo ===")
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Base features
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'experience': np.random.uniform(0, 20, n_samples),
        'education_years': np.random.normal(14, 3, n_samples),
        'purchase_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'discount': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable
    df['purchase_amount'] = (
        df['income'] * 0.1 + 
        df['experience'] * 50 + 
        df['price'] * 0.8 - 
        df['discount'] + 
        np.random.normal(0, 100, n_samples)
    )
    df['high_value_purchase'] = (df['purchase_amount'] > df['purchase_amount'].median()).astype(int)
    
    print("1. Original Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Polynomial Features
    print("\n2. Polynomial Features:")
    numerical_features = ['age', 'income', 'experience']
    X_poly, feature_names = create_polynomial_features(df[numerical_features], degree=2)
    print(f"   Original features: {len(numerical_features)}")
    print(f"   Polynomial features: {X_poly.shape[1]}")
    print(f"   First 5 feature names: {feature_names[:5]}")
    
    # Interaction Features
    print("\n3. Interaction Features:")
    feature_pairs = [('age', 'income'), ('experience', 'education_years')]
    df_interactions = create_interaction_features(df, feature_pairs)
    new_features = [col for col in df_interactions.columns if col not in df.columns]
    print(f"   Created {len(new_features)} interaction features")
    print(f"   Examples: {new_features[:3]}")
    
    # Binning Features
    print("\n4. Binning Features:")
    df_binned = binning_features(df, 'age', bins=5, labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
    binned_columns = [col for col in df_binned.columns if 'age_binned' in col]
    print(f"   Created {len(binned_columns)} binned features")
    print(f"   Examples: {binned_columns[:3]}")
    
    # Date Features
    print("\n5. Date Feature Extraction:")
    df_dates = extract_date_features(df, 'purchase_date')
    date_columns = [col for col in df_dates.columns if 'purchase_date_' in col]
    print(f"   Created {len(date_columns)} date features")
    print(f"   Examples: {date_columns[:5]}")
    
    # Domain-specific Features
    print("\n6. Domain-specific Features:")
    df_domain = domain_specific_features(df)
    domain_columns = [col for col in df_domain.columns if col not in df.columns]
    print(f"   Created {len(domain_columns)} domain-specific features")
    print(f"   Examples: {domain_columns[:5]}")

def when_to_use_feature_engineering():
    """
    Guidelines for when to use different feature engineering techniques
    """
    print("\n=== When to Use Different Feature Engineering Techniques ===")
    
    guidelines = {
        "Polynomial Features": {
            "When to use": "When you suspect non-linear relationships between features and target",
            "Benefits": "Captures non-linear patterns, can improve model fit",
            "Drawbacks": "Increases dimensionality significantly, can cause overfitting",
            "Best for": "Polynomial regression, when domain knowledge suggests non-linear relationships"
        },
        "Interaction Features": {
            "When to use": "When the effect of one feature depends on the value of another feature",
            "Benefits": "Captures feature interactions, preserves interpretability",
            "Drawbacks": "Increases feature space, requires domain knowledge",
            "Best for": "When you have hypotheses about feature interactions"
        },
        "Binning/Discretization": {
            "When to use": "When continuous variables have non-linear relationships with target",
            "Benefits": "Can capture non-linear patterns, robust to outliers",
            "Drawbacks": "Loss of information, arbitrary bin boundaries",
            "Best for": "Tree-based models, when relationships are piecewise constant"
        },
        "Date/Time Features": {
            "When to use": "When temporal patterns exist in the data",
            "Benefits": "Captures seasonality, trends, and cyclical patterns",
            "Drawbacks": "Can create many features, requires domain knowledge",
            "Best for": "Time series forecasting, when temporal patterns are important"
        },
        "Text Features": {
            "When to use": "When text data contains predictive information",
            "Benefits": "Extracts quantitative information from text",
            "Drawbacks": "Can be noisy, requires careful preprocessing",
            "Best for": "NLP tasks, when text contains relevant signals"
        },
        "Domain-specific Features": {
            "When to use": "When domain knowledge suggests meaningful derived features",
            "Benefits": "Highly interpretable, can significantly improve performance",
            "Drawbacks": "Requires domain expertise, time-consuming to develop",
            "Best for": "When you have deep understanding of the problem domain"
        },
        "Aggregate Features": {
            "When to use": "When data has natural groupings and group-level patterns are important",
            "Benefits": "Captures group-level information, can reveal hidden patterns",
            "Drawbacks": "Risk of data leakage, increases complexity",
            "Best for": "Customer analytics, when group behavior is predictive"
        }
    }
    
    for method, info in guidelines.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

def feature_importance_analysis(X, y, feature_names=None):
    """
    Analyze feature importance using different methods
    
    Args:
        X (array-like): Input features
        y (array-like): Target variable
        feature_names (list): Names of features
    
    Returns:
        dict: Dictionary containing feature importance scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    importance_scores = {}
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance_scores['Random Forest'] = rf.feature_importances_
    
    # Correlation with target
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    importance_scores['Correlation'] = correlations
    
    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    importance_scores['Mutual Information'] = mi_scores
    
    # Create summary DataFrame
    importance_df = pd.DataFrame(importance_scores, index=feature_names)
    importance_df['Average'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('Average', ascending=False)
    
    return importance_df

def automated_feature_engineering(df, target_column, max_features=50):
    """
    Automated feature engineering pipeline
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Target column name
        max_features (int): Maximum number of features to keep
    
    Returns:
        pd.DataFrame: Processed DataFrame with engineered features
    """
    df_processed = df.copy()
    
    print("=== Automated Feature Engineering Pipeline ===")
    
    # 1. Extract date features
    date_columns = df_processed.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        print(f"Processing date column: {col}")
        df_processed = extract_date_features(df_processed, col)
    
    # 2. Create interaction features for numerical columns
    numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)
    
    if len(numerical_columns) >= 2:
        print(f"Creating interaction features from {len(numerical_columns)} numerical columns")
        # Create a few interaction features
        for i in range(min(3, len(numerical_columns))):
            for j in range(i+1, min(i+3, len(numerical_columns))):
                if i != j:
                    feature_pairs = [(numerical_columns[i], numerical_columns[j])]
                    df_processed = create_interaction_features(df_processed, feature_pairs)
    
    # 3. Create binned features for high-variance numerical columns
    for col in numerical_columns[:5]:  # Limit to first 5 columns
        std_val = df_processed[col].std()
        if std_val > df_processed[col].mean() * 0.1:  # High relative variance
            print(f"Creating binned features for {col}")
            df_processed = binning_features(df_processed, col, bins=5)
    
    # 4. Domain-specific features (example for e-commerce)
    df_processed = domain_specific_features(df_processed)
    
    print(f"Final dataset shape: {df_processed.shape}")
    print(f"New features created: {df_processed.shape[1] - df.shape[1]}")
    
    return df_processed

# Example usage and testing
if __name__ == "__main__":
    # Feature engineering demo
    feature_engineering_demo()
    print("\n" + "="*50 + "\n")
    
    # When to use different techniques
    when_to_use_feature_engineering()
    print("\n" + "="*50 + "\n")
    
    # Automated feature engineering
    np.random.seed(42)
    sample_data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.uniform(0, 10, 100),
        'date_col': pd.date_range('2020-01-01', periods=100, freq='D'),
        'target': np.random.randint(0, 2, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    
    print("7. Automated Feature Engineering Example:")
    processed_df = automated_feature_engineering(sample_df, 'target')
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Various feature engineering techniques and their implementations")
    print("2. When to use different feature engineering methods")
    print("3. Automated feature engineering pipeline")
    print("4. Feature importance analysis methods")
    print("5. Domain-specific feature creation")
    print("\nKey takeaways:")
    print("- Feature engineering is crucial for model performance")
    print("- Choice of techniques depends on data type and domain knowledge")
    print("- Always validate that new features improve model performance")
    print("- Automated feature engineering can speed up the process")
    print("- Feature selection is important to avoid overfitting")