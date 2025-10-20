"""
Data Preprocessing - Feature Scaling
==============================

This module covers various feature scaling and normalization techniques that are essential
for many machine learning algorithms. Proper scaling can significantly improve model
performance and convergence speed.

Topics Covered:
- Normalization vs Standardization
- Min-Max Scaling
- Standard Scaling (Z-score normalization)
- Robust Scaling
- Unit Vector Scaling
- Custom scaling techniques
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, 
    Normalizer, PowerTransformer, QuantileTransformer
)
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

def min_max_scaling(X, feature_range=(0, 1)):
    """
    Apply Min-Max scaling to features
    
    Args:
        X (array-like): Input features
        feature_range (tuple): Range for scaled features
    
    Returns:
        array: Min-Max scaled features
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    return scaler.fit_transform(X), scaler

def standard_scaling(X):
    """
    Apply Standard scaling (Z-score normalization) to features
    
    Args:
        X (array-like): Input features
    
    Returns:
        array: Standard scaled features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def robust_scaling(X):
    """
    Apply Robust scaling using median and IQR
    
    Args:
        X (array-like): Input features
    
    Returns:
        array: Robust scaled features
    """
    scaler = RobustScaler()
    return scaler.fit_transform(X), scaler

def unit_vector_scaling(X, norm='l2'):
    """
    Apply Unit Vector scaling (Normalization)
    
    Args:
        X (array-like): Input features
        norm (str): Norm type ('l1', 'l2', 'max')
    
    Returns:
        array: Unit vector scaled features
    """
    scaler = Normalizer(norm=norm)
    return scaler.fit_transform(X), scaler

def power_transform_scaling(X, method='yeo-johnson'):
    """
    Apply Power Transform scaling to make data more Gaussian-like
    
    Args:
        X (array-like): Input features
        method (str): Transformation method ('yeo-johnson', 'box-cox')
    
    Returns:
        array: Power transformed features
    """
    scaler = PowerTransformer(method=method, standardize=True)
    return scaler.fit_transform(X), scaler

def quantile_transform_scaling(X, output_distribution='uniform'):
    """
    Apply Quantile Transform scaling
    
    Args:
        X (array-like): Input features
        output_distribution (str): Output distribution ('uniform', 'normal')
    
    Returns:
        array: Quantile transformed features
    """
    scaler = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    return scaler.fit_transform(X), scaler

def compare_scaling_methods(X, column_names=None):
    """
    Compare different scaling methods on the same data
    
    Args:
        X (array-like): Input features
        column_names (list): Column names for labeling
    
    Returns:
        dict: Dictionary containing scaled data from different methods
    """
    if column_names is None:
        column_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    results = {}
    
    # Min-Max Scaling
    X_minmax, _ = min_max_scaling(X)
    results['Min-Max'] = pd.DataFrame(X_minmax, columns=column_names)
    
    # Standard Scaling
    X_standard, _ = standard_scaling(X)
    results['Standard'] = pd.DataFrame(X_standard, columns=column_names)
    
    # Robust Scaling
    X_robust, _ = robust_scaling(X)
    results['Robust'] = pd.DataFrame(X_robust, columns=column_names)
    
    # Unit Vector Scaling
    X_unit, _ = unit_vector_scaling(X)
    results['Unit Vector'] = pd.DataFrame(X_unit, columns=column_names)
    
    # Power Transform
    try:
        X_power, _ = power_transform_scaling(X)
        results['Power Transform'] = pd.DataFrame(X_power, columns=column_names)
    except ValueError as e:
        print(f"Power Transform failed: {e}")
    
    # Quantile Transform
    X_quantile, _ = quantile_transform_scaling(X)
    results['Quantile Transform'] = pd.DataFrame(X_quantile, columns=column_names)
    
    return results

def scaling_demo():
    """
    Demonstrate feature scaling techniques
    """
    print("=== Feature Scaling Demo ===")
    
    # Create sample dataset with different scales
    np.random.seed(42)
    n_samples = 1000
    
    # Features with different scales and distributions
    data = {
        'age': np.random.normal(35, 10, n_samples),  # Normal distribution
        'income': np.random.lognormal(10, 1, n_samples),  # Log-normal distribution
        'experience': np.random.uniform(0, 30, n_samples),  # Uniform distribution
        'score': np.random.beta(2, 5, n_samples) * 100,  # Beta distribution
    }
    
    df = pd.DataFrame(data)
    X = df.values
    column_names = df.columns.tolist()
    
    print("1. Original Data Statistics:")
    print(df.describe())
    
    # Compare scaling methods
    print("\n2. Comparing Scaling Methods:")
    scaled_results = compare_scaling_methods(X, column_names)
    
    for method, scaled_df in scaled_results.items():
        print(f"\n{method} Scaling Results:")
        print(scaled_df.describe().round(3))
    
    # Demonstrate individual scaling methods
    print("\n3. Individual Scaling Method Examples:")
    
    # Min-Max Scaling
    X_minmax, minmax_scaler = min_max_scaling(X, feature_range=(0, 1))
    print(f"Min-Max Scaled Range: [{X_minmax.min():.3f}, {X_minmax.max():.3f}]")
    
    # Standard Scaling
    X_standard, standard_scaler = standard_scaling(X)
    print(f"Standard Scaled Mean: {X_standard.mean():.3f}, Std: {X_standard.std():.3f}")
    
    # Robust Scaling
    X_robust, robust_scaler = robust_scaling(X)
    print(f"Robust Scaled Median: {np.median(X_robust, axis=0)}")
    
    # Inverse transformation example
    print("\n4. Inverse Transformation Example:")
    X_original = standard_scaler.inverse_transform(X_standard[:5])
    print("Original first 5 rows:")
    print(pd.DataFrame(X[:5], columns=column_names))
    print("After standard scaling and inverse transformation:")
    print(pd.DataFrame(X_original, columns=column_names))

def visualize_scaling_effects(df):
    """
    Visualize the effects of different scaling methods
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    X = df.values
    column_names = df.columns.tolist()
    
    # Apply different scaling methods
    scaled_results = compare_scaling_methods(X, column_names)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Original data
    df.hist(bins=30, ax=axes[0])
    axes[0].set_title('Original Data')
    
    # Scaled data
    methods = list(scaled_results.keys())[:5]  # Limit to first 5 methods
    for i, method in enumerate(methods):
        scaled_results[method].hist(bins=30, ax=axes[i+1])
        axes[i+1].set_title(f'{method} Scaled Data')
    
    # Remove extra subplots
    for i in range(len(methods) + 1, 6):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def create_preprocessing_pipeline(numerical_features, categorical_features=None):
    """
    Create a preprocessing pipeline with different scalers for different features
    
    Args:
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Define transformers for numerical features
    numerical_transformers = [
        ('minmax', MinMaxScaler(), numerical_features)
    ]
    
    # If categorical features exist, add them to the pipeline
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder
        categorical_transformers = [
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
        
        # Combine numerical and categorical transformers
        preprocessor = ColumnTransformer(
            transformers= numerical_transformers + categorical_transformers,
            remainder='passthrough'
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=numerical_transformers,
            remainder='passthrough'
        )
    
    return preprocessor

def when_to_use_scaling():
    """
    Guidelines for when to use different scaling methods
    """
    print("=== When to Use Different Scaling Methods ===")
    
    guidelines = {
        "Min-Max Scaling": {
            "When to use": "When you know the approximate upper/lower bounds of your data",
            "Benefits": "Preserves original distribution shape, bounded output range",
            "Drawbacks": "Sensitive to outliers, assumes bounded data",
            "Best for": "Neural networks, KNN, PCA, when features have similar distributions"
        },
        "Standard Scaling": {
            "When to use": "When features are normally distributed or approximately normal",
            "Benefits": "Centers data around 0, preserves outliers' influence",
            "Drawbacks": "Affected by outliers, assumes normal distribution",
            "Best for": "Linear regression, logistic regression, PCA, LDA"
        },
        "Robust Scaling": {
            "When to use": "When data contains many outliers",
            "Benefits": "Robust to outliers, uses median and IQR",
            "Drawbacks": "Less interpretable, may lose some information",
            "Best for": "When outliers are important but shouldn't dominate scaling"
        },
        "Unit Vector Scaling": {
            "When to use": "When the magnitude of features is not important, only direction",
            "Benefits": "Projects data onto unit sphere, preserves relative magnitudes",
            "Drawbacks": "Can distort relationships between features",
            "Best for": "Text classification, clustering algorithms"
        },
        "Power Transform": {
            "When to use": "When features are highly skewed and need to be made more Gaussian",
            "Benefits": "Reduces skewness, makes data more normal",
            "Drawbacks": "Computationally expensive, may over-transform",
            "Best for": "When algorithms assume normality (e.g., linear models)"
        },
        "Quantile Transform": {
            "When to use": "When you want to transform data to a specific distribution",
            "Benefits": "Robust to outliers, can create uniform or normal distributions",
            "Drawbacks": "May overfit to training data, less interpretable",
            "Best for": "When you need specific distribution shapes"
        }
    }
    
    for method, info in guidelines.items():
        print(f"\n{method}:")
        for key, value in info.items():
            print(f"   {key}: {value}")

def scaling_impact_on_algorithms():
    """
    Demonstrate how scaling impacts different ML algorithms
    """
    print("\n=== Impact of Scaling on ML Algorithms ===")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Features with very different scales
    X = np.column_stack([
        np.random.normal(0, 1, n_samples),      # Feature 1: small scale
        np.random.normal(0, 1000, n_samples)    # Feature 2: large scale
    ])
    
    # Create binary target
    y = (X[:, 0] + X[:, 1]/1000 > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test algorithms with and without scaling
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    scalers = {
        'No Scaling': None,
        'Standard Scaling': StandardScaler(),
        'Min-Max Scaling': MinMaxScaler()
    }
    
    print("Algorithm Performance with Different Scaling Methods:")
    print("-" * 60)
    
    for algo_name, algorithm in algorithms.items():
        print(f"\n{algo_name}:")
        for scaler_name, scaler in scalers.items():
            if scaler is not None:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train and evaluate
            algorithm.fit(X_train_scaled, y_train)
            y_pred = algorithm.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"   {scaler_name}: {accuracy:.4f}")

# Example usage and testing
if __name__ == "__main__":
    # Feature scaling demo
    scaling_demo()
    print("\n" + "="*50 + "\n")
    
    # When to use different scaling methods
    when_to_use_scaling()
    print("\n" + "="*50 + "\n")
    
    # Impact on algorithms
    scaling_impact_on_algorithms()
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Various feature scaling techniques and their implementations")
    print("2. Comparison of different scaling methods")
    print("3. When to use each scaling method")
    print("4. Impact of scaling on machine learning algorithms")
    print("5. Creating preprocessing pipelines")
    print("6. Inverse transformation capabilities")
    print("\nKey takeaways:")
    print("- Scaling is crucial for many ML algorithms")
    print("- Choice of scaling method depends on data distribution and algorithm requirements")
    print("- Always consider the impact of outliers when choosing scaling methods")
    print("- Create reproducible preprocessing pipelines for consistent results")
    print("- Validate that scaling improves model performance")