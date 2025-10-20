"""
Advanced Data Preprocessing Pipeline
==================================

This module provides advanced data preprocessing techniques for machine learning projects.
It covers handling missing data, feature engineering, scaling, encoding, and more.

Key Components:
- Advanced missing data handling
- Feature engineering techniques
- Advanced scaling and normalization
- Categorical encoding strategies
- Outlier detection and treatment
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


class AdvancedPreprocessor:
    """
    Advanced data preprocessing pipeline for ML projects.
    
    Parameters:
    -----------
    scaling_method : str, default='standard'
        Scaling method ('standard', 'minmax', 'robust')
    encoding_method : str, default='onehot'
        Encoding method for categorical variables ('onehot', 'label', 'target')
    missing_strategy : str, default='mean'
        Strategy for handling missing values ('mean', 'median', 'knn', 'drop')
    """
    
    def __init__(self, scaling_method='standard', encoding_method='onehot', 
                 missing_strategy='mean'):
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.missing_strategy = missing_strategy
        
        # Initialize components
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.feature_selector = None
        
        # Store feature information
        self.numerical_features = []
        self.categorical_features = []
        self.feature_names = []
        
        print(f"AdvancedPreprocessor initialized with:")
        print(f"  Scaling: {scaling_method}")
        print(f"  Encoding: {encoding_method}")
        print(f"  Missing handling: {missing_strategy}")
    
    def identify_feature_types(self, X):
        """
        Identify numerical and categorical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
            
        Returns:
        --------
        self : AdvancedPreprocessor
            Returns self for method chaining
        """
        # Identify numerical features
        self.numerical_features = X.select_dtypes(
            include=[np.number, 'int', 'float', 'int64', 'float64']
        ).columns.tolist()
        
        # Identify categorical features
        self.categorical_features = X.select_dtypes(
            include=['object', 'category', 'bool']
        ).columns.tolist()
        
        self.feature_names = X.columns.tolist()
        
        print(f"Feature types identified:")
        print(f"  Numerical: {len(self.numerical_features)} features")
        print(f"  Categorical: {len(self.categorical_features)} features")
        
        return self
    
    def handle_missing_values(self, X):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with potential missing values
            
        Returns:
        --------
        X_processed : pandas.DataFrame
            Data with missing values handled
        """
        X_processed = X.copy()
        
        if self.missing_strategy == 'drop':
            # Drop rows with missing values
            X_processed = X_processed.dropna()
            print(f"Dropped rows with missing values. New shape: {X_processed.shape}")
            
        elif self.missing_strategy == 'mean':
            # Impute numerical features with mean, categorical with mode
            if self.numerical_features:
                num_imputer = SimpleImputer(strategy='mean')
                X_processed[self.numerical_features] = num_imputer.fit_transform(
                    X_processed[self.numerical_features]
                )
            
            if self.categorical_features:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_processed[self.categorical_features] = cat_imputer.fit_transform(
                    X_processed[self.categorical_features]
                )
            
            print("Missing values imputed with mean/mode")
            
        elif self.missing_strategy == 'median':
            # Impute numerical features with median, categorical with mode
            if self.numerical_features:
                num_imputer = SimpleImputer(strategy='median')
                X_processed[self.numerical_features] = num_imputer.fit_transform(
                    X_processed[self.numerical_features]
                )
            
            if self.categorical_features:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_processed[self.categorical_features] = cat_imputer.fit_transform(
                    X_processed[self.categorical_features]
                )
            
            print("Missing values imputed with median/mode")
            
        elif self.missing_strategy == 'knn':
            # Use KNN imputation for all features
            if len(self.numerical_features) > 0:
                knn_imputer = KNNImputer(n_neighbors=5)
                X_processed[self.numerical_features] = knn_imputer.fit_transform(
                    X_processed[self.numerical_features]
                )
                print("Missing values imputed with KNN")
        
        return X_processed
    
    def encode_categorical_features(self, X, y=None):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with categorical features
        y : pandas.Series, optional
            Target variable for target encoding
            
        Returns:
        --------
        X_encoded : pandas.DataFrame
            Data with encoded categorical features
        """
        X_encoded = X.copy()
        
        if not self.categorical_features:
            print("No categorical features to encode")
            return X_encoded
        
        if self.encoding_method == 'label':
            # Label encoding
            for feature in self.categorical_features:
                le = LabelEncoder()
                X_encoded[feature] = le.fit_transform(X_encoded[feature].astype(str))
                self.encoders[feature] = le
            print("Categorical features label encoded")
            
        elif self.encoding_method == 'onehot':
            # One-hot encoding
            X_encoded = pd.get_dummies(X_encoded, columns=self.categorical_features, 
                                     prefix=self.categorical_features)
            print("Categorical features one-hot encoded")
            
        elif self.encoding_method == 'target' and y is not None:
            # Target encoding (mean encoding)
            for feature in self.categorical_features:
                # Calculate mean target value for each category
                target_means = y.groupby(X[feature]).mean()
                X_encoded[feature] = X[feature].map(target_means)
                self.encoders[feature] = target_means
            print("Categorical features target encoded")
        
        return X_encoded
    
    def scale_features(self, X):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data with numerical features
            
        Returns:
        --------
        X_scaled : pandas.DataFrame
            Data with scaled numerical features
        """
        X_scaled = X.copy()
        
        if not self.numerical_features:
            print("No numerical features to scale")
            return X_scaled
        
        # Select scaler
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {self.scaling_method}")
        
        # Scale numerical features
        X_scaled[self.numerical_features] = self.scaler.fit_transform(
            X_scaled[self.numerical_features]
        )
        
        print(f"Numerical features scaled using {self.scaling_method} scaling")
        return X_scaled
    
    def detect_outliers(self, X, method='iqr', threshold=1.5):
        """
        Detect outliers in numerical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        method : str, default='iqr'
            Outlier detection method ('iqr', 'zscore', 'isolation')
        threshold : float, default=1.5
            Threshold for outlier detection
            
        Returns:
        --------
        outliers : dict
            Dictionary with outlier information for each feature
        """
        outliers = {}
        
        for feature in self.numerical_features:
            if method == 'iqr':
                # Interquartile Range method
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (X[feature] < lower_bound) | (X[feature] > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(X[feature]))
                outlier_mask = z_scores > threshold
                
            outliers[feature] = {
                'count': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(X)) * 100,
                'indices': X[outlier_mask].index.tolist()
            }
        
        # Print summary
        print("Outlier Detection Results:")
        print("=" * 30)
        for feature, info in outliers.items():
            print(f"{feature}: {info['count']} outliers ({info['percentage']:.2f}%)")
        
        return outliers
    
    def handle_outliers(self, X, method='cap', outlier_info=None):
        """
        Handle outliers in numerical features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        method : str, default='cap'
            Outlier handling method ('cap', 'remove', 'transform')
        outlier_info : dict, optional
            Pre-computed outlier information
            
        Returns:
        --------
        X_processed : pandas.DataFrame
            Data with outliers handled
        """
        X_processed = X.copy()
        
        if outlier_info is None:
            outlier_info = self.detect_outliers(X_processed)
        
        for feature in self.numerical_features:
            if outlier_info[feature]['count'] == 0:
                continue
                
            if method == 'cap':
                # Cap outliers at bounds
                Q1 = X_processed[feature].quantile(0.25)
                Q3 = X_processed[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                X_processed[feature] = np.clip(
                    X_processed[feature], lower_bound, upper_bound
                )
                print(f"Outliers capped for {feature}")
                
            elif method == 'remove':
                # Remove outlier rows
                outlier_indices = outlier_info[feature]['indices']
                X_processed = X_processed.drop(outlier_indices)
                print(f"Outliers removed for {feature}")
                
            elif method == 'transform':
                # Apply log transformation
                X_processed[feature] = np.log1p(X_processed[feature] - X_processed[feature].min())
                print(f"Outliers transformed for {feature}")
        
        return X_processed
    
    def create_polynomial_features(self, X, degree=2, interaction_only=False):
        """
        Create polynomial and interaction features.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        degree : int, default=2
            Degree of polynomial features
        interaction_only : bool, default=False
            Whether to only create interaction features
            
        Returns:
        --------
        X_poly : pandas.DataFrame
            Data with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        if not self.numerical_features:
            print("No numerical features for polynomial creation")
            return X
        
        # Create polynomial features for numerical features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                include_bias=False)
        
        X_poly_features = poly.fit_transform(X[self.numerical_features])
        feature_names = poly.get_feature_names_out(self.numerical_features)
        
        # Create DataFrame with polynomial features
        X_poly = pd.DataFrame(X_poly_features, columns=feature_names, index=X.index)
        
        # Combine with categorical features
        if self.categorical_features:
            X_poly = pd.concat([X_poly, X[self.categorical_features]], axis=1)
        
        print(f"Polynomial features created: {X_poly.shape[1]} total features")
        return X_poly
    
    def select_features(self, X, y, k=10, method='univariate'):
        """
        Select top k features based on importance.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        y : pandas.Series
            Target variable
        k : int, default=10
            Number of features to select
        method : str, default='univariate'
            Feature selection method ('univariate', 'recursive', 'importance')
            
        Returns:
        --------
        X_selected : pandas.DataFrame
            Data with selected features
        """
        if method == 'univariate':
            # Univariate feature selection
            if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number):
                # Regression
                selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            else:
                # Classification
                selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            print(f"Selected {len(selected_features)} features using univariate selection")
            print(f"Selected features: {selected_features}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        else:
            print("Other feature selection methods not implemented in this example")
            return X
    
    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Input data
        y : pandas.Series, optional
            Target variable
            
        Returns:
        --------
        X_processed : pandas.DataFrame
            Preprocessed data
        """
        print("Starting preprocessing pipeline...")
        print("=" * 35)
        
        # Step 1: Identify feature types
        self.identify_feature_types(X)
        
        # Step 2: Handle missing values
        X_processed = self.handle_missing_values(X)
        
        # Step 3: Encode categorical features
        X_processed = self.encode_categorical_features(X_processed, y)
        
        # Step 4: Scale features
        X_processed = self.scale_features(X_processed)
        
        print("Preprocessing pipeline completed")
        print(f"Final data shape: {X_processed.shape}")
        
        return X_processed
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            New data to transform
            
        Returns:
        --------
        X_transformed : pandas.DataFrame
            Transformed data
        """
        # Apply same transformations as in fit_transform
        # This is a simplified version - in practice, you would apply
        # the fitted components (scaler, encoders, etc.)
        print("Transforming new data...")
        return X  # Placeholder implementation


# Feature Engineering Utilities
class FeatureEngineer:
    """
    Advanced feature engineering utilities.
    """
    
    @staticmethod
    def create_time_features(df, date_column):
        """
        Create time-based features from date column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        date_column : str
            Name of date column
            
        Returns:
        --------
        df_with_time : pandas.DataFrame
            Data with time features
        """
        df_with_time = df.copy()
        df_with_time[date_column] = pd.to_datetime(df_with_time[date_column])
        
        # Extract time components
        df_with_time[f'{date_column}_year'] = df_with_time[date_column].dt.year
        df_with_time[f'{date_column}_month'] = df_with_time[date_column].dt.month
        df_with_time[f'{date_column}_day'] = df_with_time[date_column].dt.day
        df_with_time[f'{date_column}_dayofweek'] = df_with_time[date_column].dt.dayofweek
        df_with_time[f'{date_column}_quarter'] = df_with_time[date_column].dt.quarter
        
        return df_with_time
    
    @staticmethod
    def create_aggregate_features(df, group_column, agg_columns, agg_funcs=['mean', 'std']):
        """
        Create aggregate features by grouping.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data
        group_column : str
            Column to group by
        agg_columns : list
            Columns to aggregate
        agg_funcs : list, default=['mean', 'std']
            Aggregation functions
            
        Returns:
        --------
        df_with_agg : pandas.DataFrame
            Data with aggregate features
        """
        df_with_agg = df.copy()
        
        # Create aggregate features
        for col in agg_columns:
            for func in agg_funcs:
                agg_feature = df.groupby(group_column)[col].agg(func)
                df_with_agg[f'{col}_{func}_by_{group_column}'] = df_with_agg[group_column].map(agg_feature)
        
        return df_with_agg


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Advanced Data Preprocessing Pipeline Demonstration")
    print("=" * 52)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample dataset
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'satisfaction': np.random.randint(1, 11, n_samples)
    })
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    sample_data.loc[missing_indices[:25], 'income'] = np.nan
    sample_data.loc[missing_indices[25:], 'education'] = np.nan
    
    # Introduce some outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    sample_data.loc[outlier_indices[:10], 'income'] += 100000
    sample_data.loc[outlier_indices[10:], 'age'] += 50
    
    print(f"Sample data created: {sample_data.shape}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
    
    # Initialize preprocessor
    preprocessor = AdvancedPreprocessor(
        scaling_method='standard',
        encoding_method='onehot',
        missing_strategy='mean'
    )
    
    # Fit and transform data
    processed_data = preprocessor.fit_transform(sample_data, sample_data['satisfaction'])
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Numerical features: {len(preprocessor.numerical_features)}")
    print(f"Categorical features: {len(preprocessor.categorical_features)}")
    
    # Demonstrate outlier detection
    print("\nOutlier Detection:")
    outlier_info = preprocessor.detect_outliers(sample_data)
    
    # Demonstrate feature engineering
    print("\nFeature Engineering:")
    engineer = FeatureEngineer()
    
    # Create time features (simulated)
    time_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    })
    time_features = engineer.create_time_features(time_data, 'date')
    print(f"Time features created: {time_features.shape}")
    
    # Create aggregate features
    agg_data = engineer.create_aggregate_features(
        sample_data, 'city', ['age', 'income'], ['mean', 'std']
    )
    print(f"Aggregate features created: {agg_data.shape}")
    
    # Advanced preprocessing pipeline summary
    print("\n" + "="*50)
    print("Advanced Preprocessing Pipeline Summary")
    print("="*50)
    print("1. Missing Data Handling:")
    print("   - Mean/Median imputation")
    print("   - KNN imputation")
    print("   - Drop missing rows")
    print("   - Advanced imputation strategies")
    
    print("\n2. Feature Encoding:")
    print("   - One-hot encoding")
    print("   - Label encoding")
    print("   - Target encoding")
    print("   - Embedding-based encoding")
    
    print("\n3. Feature Scaling:")
    print("   - Standard scaling (Z-score)")
    print("   - Min-Max scaling")
    print("   - Robust scaling")
    print("   - Quantile transformation")
    
    print("\n4. Outlier Treatment:")
    print("   - IQR-based detection")
    print("   - Z-score detection")
    print("   - Isolation Forest")
    print("   - Capping and flooring")
    
    print("\n5. Feature Engineering:")
    print("   - Polynomial features")
    print("   - Interaction terms")
    print("   - Time-based features")
    print("   - Domain-specific features")
    
    print("\n6. Feature Selection:")
    print("   - Univariate selection")
    print("   - Recursive feature elimination")
    print("   - Model-based importance")
    print("   - Correlation-based selection")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Data Preprocessing")
    print("="*50)
    print("1. Data Quality:")
    print("   - Validate data integrity")
    print("   - Handle data drift")
    print("   - Ensure consistency")
    print("   - Document data sources")
    
    print("\n2. Preprocessing Strategy:")
    print("   - Apply same transformations to train/test")
    print("   - Avoid data leakage")
    print("   - Use domain knowledge")
    print("   - Validate preprocessing steps")
    
    print("\n3. Feature Engineering:")
    print("   - Create meaningful features")
    print("   - Avoid over-engineering")
    print("   - Consider computational cost")
    print("   - Test feature importance")
    
    print("\n4. Scalability:")
    print("   - Optimize for large datasets")
    print("   - Use parallel processing")
    print("   - Implement caching")
    print("   - Monitor memory usage")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using:")
    print("- Scikit-learn Pipelines: For chaining preprocessing steps")
    print("- Feature-engine: Advanced feature engineering library")
    print("- Category Encoders: Advanced categorical encoding")
    print("- These provide optimized and battle-tested implementations")