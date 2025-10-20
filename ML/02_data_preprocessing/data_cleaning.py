"""
Data Preprocessing - Data Cleaning
============================

This module covers techniques for cleaning and handling missing data in machine learning datasets.
Proper data cleaning is essential for building robust and accurate models.

Topics Covered:
- Identifying missing data
- Handling missing values
- Outlier detection and treatment
- Data validation and quality checks
- Real-world data cleaning scenarios
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def identify_missing_data(df):
    """
    Identify and analyze missing data in a DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: Summary of missing data
    """
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percentage = 100 * missing_count / len(df)
    
    # Create summary DataFrame
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage
    })
    
    # Filter out columns with no missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
    
    return missing_df

def handle_missing_numerical(df, column, strategy='mean'):
    """
    Handle missing values in numerical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        strategy (str): Imputation strategy ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'mean':
        df_copy[column].fillna(df_copy[column].mean(), inplace=True)
    elif strategy == 'median':
        df_copy[column].fillna(df_copy[column].median(), inplace=True)
    elif strategy == 'mode':
        df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
    elif strategy == 'drop':
        df_copy.dropna(subset=[column], inplace=True)
    elif strategy == 'forward_fill':
        df_copy[column].fillna(method='ffill', inplace=True)
    elif strategy == 'backward_fill':
        df_copy[column].fillna(method='bfill', inplace=True)
    
    return df_copy

def handle_missing_categorical(df, column, strategy='mode'):
    """
    Handle missing values in categorical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        strategy (str): Imputation strategy ('mode', 'missing_category', 'drop')
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'mode':
        df_copy[column].fillna(df_copy[column].mode()[0], inplace=True)
    elif strategy == 'missing_category':
        df_copy[column].fillna('Missing', inplace=True)
    elif strategy == 'drop':
        df_copy.dropna(subset=[column], inplace=True)
    
    return df_copy

def detect_outliers_iqr(df, column, factor=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        factor (float): IQR factor for outlier detection
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        threshold (float): Z-score threshold for outlier detection
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > threshold

def handle_outliers(df, column, method='cap', factor=1.5):
    """
    Handle outliers in numerical columns
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name
        method (str): Outlier handling method ('cap', 'remove', 'transform')
        factor (float): IQR factor for capping
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    df_copy = df.copy()
    
    if method == 'cap':
        Q1 = df_copy[column].quantile(0.25)
        Q3 = df_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        df_copy[column] = np.clip(df_copy[column], lower_bound, upper_bound)
    
    elif method == 'remove':
        outliers = detect_outliers_iqr(df_copy, column, factor)
        df_copy = df_copy[~outliers]
    
    elif method == 'transform':
        # Log transformation for positive values
        if (df_copy[column] > 0).all():
            df_copy[column] = np.log1p(df_copy[column])
    
    return df_copy

def advanced_imputation(df, numerical_columns, categorical_columns):
    """
    Advanced imputation techniques for mixed data types
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_columns (list): List of numerical column names
        categorical_columns (list): List of categorical column names
    
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    df_copy = df.copy()
    
    # Impute numerical columns using KNN
    if numerical_columns:
        knn_imputer = KNNImputer(n_neighbors=5)
        df_copy[numerical_columns] = knn_imputer.fit_transform(df_copy[numerical_columns])
    
    # Impute categorical columns using mode
    for col in categorical_columns:
        if df_copy[col].isnull().any():
            mode_value = df_copy[col].mode()
            if not mode_value.empty:
                df_copy[col].fillna(mode_value[0], inplace=True)
    
    return df_copy

def data_cleaning_demo():
    """
    Demonstrate data cleaning techniques
    """
    print("=== Data Cleaning Demo ===")
    
    # Create sample dataset with missing values and outliers
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'satisfaction': np.random.uniform(1, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices[:20], 'age'] = np.nan
    df.loc[missing_indices[20:40], 'income'] = np.nan
    df.loc[missing_indices[40:], 'education'] = np.nan
    
    # Introduce outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices[:10], 'income'] *= 10  # Extreme high incomes
    df.loc[outlier_indices[10:], 'age'] = np.random.uniform(100, 150, 10)  # Extreme ages
    
    print("1. Original Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values:\n{df.isnull().sum()}")
    
    # Identify missing data
    print("\n2. Missing Data Analysis:")
    missing_summary = identify_missing_data(df)
    print(missing_summary)
    
    # Handle missing numerical data
    print("\n3. Handling Missing Numerical Data:")
    df_cleaned = handle_missing_numerical(df, 'age', 'median')
    df_cleaned = handle_missing_numerical(df_cleaned, 'income', 'median')
    print(f"   Missing age values after imputation: {df_cleaned['age'].isnull().sum()}")
    print(f"   Missing income values after imputation: {df_cleaned['income'].isnull().sum()}")
    
    # Handle missing categorical data
    print("\n4. Handling Missing Categorical Data:")
    df_cleaned = handle_missing_categorical(df_cleaned, 'education', 'mode')
    print(f"   Missing education values after imputation: {df_cleaned['education'].isnull().sum()}")
    
    # Detect outliers
    print("\n5. Outlier Detection:")
    age_outliers = detect_outliers_iqr(df_cleaned, 'age')
    income_outliers = detect_outliers_iqr(df_cleaned, 'income')
    print(f"   Age outliers: {age_outliers.sum()}")
    print(f"   Income outliers: {income_outliers.sum()}")
    
    # Handle outliers
    print("\n6. Outlier Handling:")
    df_final = handle_outliers(df_cleaned, 'income', 'cap')
    df_final = handle_outliers(df_final, 'age', 'cap')
    print(f"   Income range after capping: {df_final['income'].min():.2f} - {df_final['income'].max():.2f}")
    print(f"   Age range after capping: {df_final['age'].min():.2f} - {df_final['age'].max():.2f}")
    
    print(f"\n7. Final Dataset Shape: {df_final.shape}")
    print(f"   Final missing values:\n{df_final.isnull().sum()}")

def data_quality_checks(df):
    """
    Perform comprehensive data quality checks
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        dict: Dictionary containing quality check results
    """
    quality_report = {}
    
    # Basic info
    quality_report['shape'] = df.shape
    quality_report['columns'] = list(df.columns)
    quality_report['dtypes'] = df.dtypes.to_dict()
    
    # Missing data
    quality_report['missing_count'] = df.isnull().sum().to_dict()
    quality_report['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
    
    # Duplicates
    quality_report['duplicate_rows'] = df.duplicated().sum()
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['numerical_columns'] = {}
    for col in numerical_cols:
        quality_report['numerical_columns'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'unique_values': df[col].nunique()
        }
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    quality_report['categorical_columns'] = {}
    for col in categorical_cols:
        quality_report['categorical_columns'][col] = {
            'unique_values': df[col].nunique(),
            'top_categories': df[col].value_counts().head().to_dict()
        }
    
    return quality_report

def visualize_data_quality(df):
    """
    Create visualizations for data quality assessment
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Missing data heatmap
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, ax=axes[0,0])
    axes[0,0].set_title('Missing Data Pattern')
    
    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns[:4]
    for i, col in enumerate(numerical_cols):
        if i < 2:
            df[col].hist(bins=30, ax=axes[0,1] if i == 0 else axes[1,0])
            (axes[0,1] if i == 0 else axes[1,0]).set_title(f'Distribution of {col}')
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1,1])
        axes[1,1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Data cleaning demo
    data_cleaning_demo()
    print("\n" + "="*50 + "\n")
    
    # Create a sample dataset for quality checks
    np.random.seed(42)
    sample_data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.uniform(0, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Add some missing values for demonstration
    sample_df.loc[5:10, 'feature1'] = np.nan
    sample_df.loc[15:20, 'category'] = np.nan
    
    print("=== Data Quality Assessment ===")
    quality_report = data_quality_checks(sample_df)
    print("Dataset Shape:", quality_report['shape'])
    print("Missing Percentages:")
    for col, pct in quality_report['missing_percentage'].items():
        if pct > 0:
            print(f"  {col}: {pct:.2f}%")
    print("Duplicate Rows:", quality_report['duplicate_rows'])
    
    print("\n=== Summary ===")
    print("This example demonstrated:")
    print("1. Techniques for identifying missing data")
    print("2. Methods for handling missing values in numerical and categorical data")
    print("3. Outlier detection and treatment approaches")
    print("4. Advanced imputation techniques")
    print("5. Comprehensive data quality assessment")
    print("6. Data visualization for quality analysis")
    print("\nKey takeaways:")
    print("- Always assess data quality before modeling")
    print("- Choose appropriate imputation methods based on data characteristics")
    print("- Handle outliers carefully to avoid losing important information")
    print("- Create reproducible preprocessing pipelines")
    print("- Validate data quality after preprocessing steps")