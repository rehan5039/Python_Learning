"""
Data Preprocessing - Practice Problems
===============================

This file contains practice problems for data preprocessing techniques with solutions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Problem 1: Data Cleaning Practice
def problem_1():
    """
    Data cleaning practice problems:
    """
    
    print("Problem 1: Data Cleaning Practice")
    print("=" * 35)
    
    # 1. Handle missing values in different scenarios
    def handle_missing_values_scenario():
        """Handle missing values with different strategies"""
        # Create dataset with various missing value patterns
        data = {
            'age': [25, np.nan, 30, 35, np.nan, 40, 45, np.nan],
            'income': [50000, 60000, np.nan, 80000, 90000, np.nan, 110000, 120000],
            'education': ['Bachelor', 'Master', np.nan, 'PhD', 'Bachelor', 'Master', np.nan, 'PhD'],
            'experience': [2, 5, np.nan, 10, 12, 15, 20, np.nan],
            'department': ['IT', 'HR', 'Finance', 'IT', np.nan, 'HR', 'Finance', 'IT']
        }
        
        df = pd.DataFrame(data)
        print("Original Data:")
        print(df)
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing numerical values
        df_filled = df.copy()
        
        # Age: Use median (robust to outliers)
        df_filled['age'].fillna(df_filled['age'].median(), inplace=True)
        
        # Income: Use mean
        df_filled['income'].fillna(df_filled['income'].mean(), inplace=True)
        
        # Experience: Forward fill
        df_filled['experience'].fillna(method='ffill', inplace=True)
        
        # Handle missing categorical values
        # Education: Use mode
        df_filled['education'].fillna(df_filled['education'].mode()[0], inplace=True)
        
        # Department: Create new category
        df_filled['department'].fillna('Unknown', inplace=True)
        
        print("\nAfter handling missing values:")
        print(df_filled)
        print(f"Missing values:\n{df_filled.isnull().sum()}")
        
        return df_filled
    
    # 2. Detect and treat outliers
    def detect_and_treat_outliers():
        """Detect and treat outliers in numerical data"""
        # Create dataset with outliers
        np.random.seed(42)
        ages = np.random.normal(35, 10, 100)
        incomes = np.random.lognormal(10, 0.5, 100)
        
        # Introduce outliers
        ages[0] = 150  # Extreme age
        incomes[0] = 1000000  # Extreme income
        
        df = pd.DataFrame({'age': ages, 'income': incomes})
        
        print("\nOriginal Data Statistics:")
        print(df.describe())
        
        # Detect outliers using IQR method
        def detect_outliers_iqr(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        age_outliers = detect_outliers_iqr(df['age'])
        income_outliers = detect_outliers_iqr(df['income'])
        
        print(f"\nOutliers detected:")
        print(f"Age outliers: {age_outliers.sum()}")
        print(f"Income outliers: {income_outliers.sum()}")
        
        # Treat outliers by capping
        def cap_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return np.clip(series, lower_bound, upper_bound)
        
        df_treated = df.copy()
        df_treated['age'] = cap_outliers(df_treated['age'])
        df_treated['income'] = cap_outliers(df_treated['income'])
        
        print("\nAfter treating outliers:")
        print(df_treated.describe())
        
        return df_treated
    
    # Run the problems
    print("1. Handling Missing Values:")
    df_cleaned = handle_missing_values_scenario()
    
    print("\n2. Detecting and Treating Outliers:")
    df_treated = detect_and_treat_outliers()

# Problem 2: Feature Scaling Exercises
def problem_2():
    """
    Feature scaling practice problems:
    """
    
    print("\nProblem 2: Feature Scaling Exercises")
    print("=" * 35)
    
    # 1. Compare different scaling methods
    def compare_scaling_methods():
        """Compare different feature scaling methods"""
        # Create dataset with different scales
        np.random.seed(42)
        data = {
            'age': np.random.normal(35, 10, 1000),  # Small scale
            'income': np.random.lognormal(10, 1, 1000),  # Large scale, skewed
            'score': np.random.uniform(0, 100, 1000),  # Uniform distribution
            'experience': np.random.exponential(5, 1000)  # Exponential distribution
        }
        
        df = pd.DataFrame(data)
        X = df.values
        
        print("Original Data Statistics:")
        print(df.describe())
        
        # Apply different scaling methods
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        # Min-Max Scaling
        minmax_scaler = MinMaxScaler()
        X_minmax = minmax_scaler.fit_transform(X)
        df_minmax = pd.DataFrame(X_minmax, columns=df.columns)
        
        # Standard Scaling
        standard_scaler = StandardScaler()
        X_standard = standard_scaler.fit_transform(X)
        df_standard = pd.DataFrame(X_standard, columns=df.columns)
        
        # Robust Scaling
        robust_scaler = RobustScaler()
        X_robust = robust_scaler.fit_transform(X)
        df_robust = pd.DataFrame(X_robust, columns=df.columns)
        
        print("\nMin-Max Scaled Data Statistics:")
        print(df_minmax.describe())
        
        print("\nStandard Scaled Data Statistics:")
        print(df_standard.describe())
        
        print("\nRobust Scaled Data Statistics:")
        print(df_robust.describe())
        
        return {
            'original': df,
            'minmax': df_minmax,
            'standard': df_standard,
            'robust': df_robust
        }
    
    # 2. Choose appropriate scaling for different scenarios
    def choose_scaling_scenario():
        """Choose appropriate scaling method for different scenarios"""
        scenarios = {
            "Neural Network": {
                "data_characteristics": "Features with very different scales",
                "recommended_scaling": "Min-Max Scaling (0,1)",
                "reason": "Neural networks are sensitive to input scale, bounded range helps with convergence"
            },
            "Linear Regression with Outliers": {
                "data_characteristics": "Normally distributed features with some outliers",
                "recommended_scaling": "Robust Scaling",
                "reason": "Robust to outliers, uses median and IQR instead of mean and std"
            },
            "K-Means Clustering": {
                "data_characteristics": "Features with different units and scales",
                "recommended_scaling": "Standard Scaling",
                "reason": "Distance-based algorithm, needs features to contribute equally"
            },
            "Decision Tree": {
                "data_characteristics": "Mixed numerical and categorical features",
                "recommended_scaling": "No scaling needed",
                "reason": "Tree-based algorithms are scale-invariant"
            }
        }
        
        print("Scaling Method Recommendations:")
        for scenario, details in scenarios.items():
            print(f"\n{scenario}:")
            print(f"  Data Characteristics: {details['data_characteristics']}")
            print(f"  Recommended Scaling: {details['recommended_scaling']}")
            print(f"  Reason: {details['reason']}")
    
    # Run the problems
    print("1. Comparing Scaling Methods:")
    scaling_results = compare_scaling_methods()
    
    print("\n2. Choosing Appropriate Scaling:")
    choose_scaling_scenario()

# Problem 3: Encoding Challenges
def problem_3():
    """
    Encoding practice problems:
    """
    
    print("\nProblem 3: Encoding Challenges")
    print("=" * 30)
    
    # 1. Encode categorical variables with different methods
    def encode_categorical_variables():
        """Encode categorical variables using different methods"""
        # Create dataset with categorical variables
        np.random.seed(42)
        data = {
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 100),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
            'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR'], 100),
            'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract'], 100)
        }
        
        df = pd.DataFrame(data)
        
        print("Original Categorical Data:")
        for col in df.columns:
            print(f"{col}: {df[col].unique()}")
        
        # Label Encoding
        df_label = df.copy()
        label_encoders = {}
        for col in df.columns:
            le = LabelEncoder()
            df_label[f'{col}_label'] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        print("\nAfter Label Encoding:")
        for col in df.columns:
            print(f"{col}: {dict(zip(label_encoders[col].classes_, range(len(label_encoders[col].classes_))))}")
        
        # One-Hot Encoding
        df_onehot = pd.get_dummies(df, prefix=df.columns)
        print(f"\nAfter One-Hot Encoding:")
        print(f"Shape: {df_onehot.shape}")
        print(f"First 5 columns: {df_onehot.columns[:5].tolist()}")
        
        return df_label, df_onehot
    
    # 2. Handle high cardinality categorical variables
    def handle_high_cardinality():
        """Handle high cardinality categorical variables"""
        # Create dataset with high cardinality
        np.random.seed(42)
        cities = [f'City_{i}' for i in range(1, 101)]  # 100 unique cities
        data = {
            'user_id': range(1000),
            'city': np.random.choice(cities, 1000),
            'age': np.random.randint(18, 80, 1000)
        }
        
        df = pd.DataFrame(data)
        
        print("High Cardinality Data:")
        print(f"Unique cities: {df['city'].nunique()}")
        print(f"Top 10 cities by frequency:")
        print(df['city'].value_counts().head(10))
        
        # Method 1: Group rare categories
        city_counts = df['city'].value_counts()
        top_cities = city_counts.head(10).index
        df['city_grouped'] = df['city'].apply(lambda x: x if x in top_cities else 'Other')
        
        print(f"\nAfter grouping rare categories:")
        print(f"Unique cities: {df['city_grouped'].nunique()}")
        print(f"Categories: {df['city_grouped'].unique()}")
        
        # Method 2: Frequency encoding
        city_freq = df['city'].value_counts().to_dict()
        df['city_frequency'] = df['city'].map(city_freq)
        
        print(f"\nFrequency encoding (first 10 values):")
        print(df[['city', 'city_frequency']].head(10))
        
        return df
    
    # Run the problems
    print("1. Encoding Categorical Variables:")
    df_label, df_onehot = encode_categorical_variables()
    
    print("\n2. Handling High Cardinality:")
    df_high_card = handle_high_cardinality()

# Problem 4: Feature Engineering Problems
def problem_4():
    """
    Feature engineering practice problems:
    """
    
    print("\nProblem 4: Feature Engineering Problems")
    print("=" * 35)
    
    # 1. Create polynomial and interaction features
    def create_polynomial_interaction_features():
        """Create polynomial and interaction features"""
        # Create dataset
        np.random.seed(42)
        data = {
            'x1': np.random.normal(0, 1, 100),
            'x2': np.random.normal(0, 1, 100),
            'x3': np.random.normal(0, 1, 100)
        }
        
        df = pd.DataFrame(data)
        
        print("Original Features:")
        print(df.head())
        
        # Create polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        
        # Degree 2 polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(df)
        feature_names = poly.get_feature_names_out(df.columns)
        df_poly = pd.DataFrame(X_poly, columns=feature_names)
        
        print(f"\nPolynomial Features (degree 2):")
        print(f"Original features: {df.shape[1]}")
        print(f"Polynomial features: {df_poly.shape[1]}")
        print(f"Feature names: {feature_names}")
        
        # Create specific interaction features
        df_interact = df.copy()
        df_interact['x1_x_x2'] = df['x1'] * df['x2']
        df_interact['x1_x_x3'] = df['x1'] * df['x3']
        df_interact['x2_x_x3'] = df['x2'] * df['x3']
        
        print(f"\nSpecific Interaction Features:")
        print(df_interact[['x1_x_x2', 'x1_x_x3', 'x2_x_x3']].head())
        
        return df_poly, df_interact
    
    # 2. Extract features from date/time data
    def extract_datetime_features():
        """Extract features from date/time data"""
        # Create dataset with datetime
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        np.random.seed(42)
        data = {
            'date': dates,
            'sales': np.random.poisson(100, 365)
        }
        
        df = pd.DataFrame(data)
        
        print("Original DateTime Data:")
        print(df.head())
        
        # Extract date components
        df_features = df.copy()
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['day'] = df_features['date'].dt.day
        df_features['dayofweek'] = df_features['date'].dt.dayofweek
        df_features['quarter'] = df_features['date'].dt.quarter
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        df_features['is_month_start'] = (df_features['day'] <= 7).astype(int)
        df_features['is_month_end'] = (df_features['day'] >= 25).astype(int)
        
        # Cyclical features
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        print(f"\nExtracted DateTime Features:")
        datetime_columns = [col for col in df_features.columns if col != 'date' and col != 'sales']
        print(f"Created {len(datetime_columns)} features:")
        print(datetime_columns)
        
        print(f"\nSample of extracted features:")
        print(df_features[datetime_columns].head())
        
        return df_features
    
    # Run the problems
    print("1. Polynomial and Interaction Features:")
    df_poly, df_interact = create_polynomial_interaction_features()
    
    print("\n2. DateTime Feature Extraction:")
    df_datetime = extract_datetime_features()

# Problem 5: Pipeline Development
def problem_5():
    """
    Pipeline development practice problems:
    """
    
    print("\nProblem 5: Pipeline Development")
    print("=" * 30)
    
    # 1. Build comprehensive preprocessing pipeline
    def build_preprocessing_pipeline():
        """Build comprehensive preprocessing pipeline"""
        # Create sample dataset
        np.random.seed(42)
        data = {
            'age': np.random.normal(35, 10, 1000),
            'income': np.random.lognormal(10, 0.5, 1000),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 1000),
            'experience': np.random.uniform(0, 20, 1000)
        }
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        missing_indices = np.random.choice(df.index, size=50, replace=False)
        df.loc[missing_indices[:25], 'age'] = np.nan
        df.loc[missing_indices[25:], 'income'] = np.nan
        
        # Introduce outliers
        outlier_indices = np.random.choice(df.index, size=10, replace=False)
        df.loc[outlier_indices[:5], 'income'] *= 5
        
        print("Dataset with missing values and outliers:")
        print(f"Shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Create preprocessing pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        # Define column types
        numerical_features = ['age', 'income', 'experience']
        categorical_features = ['education', 'city']
        
        # Numerical preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        print(f"\nPreprocessing Pipeline Created:")
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Apply preprocessing
        X = df.dropna()  # For simplicity, drop rows with missing target
        X_preprocessed = preprocessor.fit_transform(X)
        
        print(f"\nAfter preprocessing:")
        print(f"Original shape: {X.shape}")
        print(f"Preprocessed shape: {X_preprocessed.shape}")
        
        return preprocessor, X_preprocessed
    
    # 2. Handle data leakage in preprocessing
    def handle_data_leakage():
        """Handle data leakage in preprocessing"""
        # Create dataset
        np.random.seed(42)
        data = {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'target': np.random.randint(0, 2, 1000)
        }
        
        df = pd.DataFrame(data)
        
        # Split data properly
        X = df[['feature1', 'feature2']]
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Proper Train-Test Split:")
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # WRONG WAY (data leakage)
        print("\n❌ WRONG WAY - Data Leakage:")
        print("Fitting scaler on entire dataset before splitting")
        scaler_wrong = StandardScaler()
        X_scaled_wrong = scaler_wrong.fit_transform(X)  # WRONG - uses test data
        X_train_wrong, X_test_wrong, y_train_wrong, y_test_wrong = train_test_split(
            X_scaled_wrong, y, test_size=0.2, random_state=42)
        print("This approach uses test data information during training!")
        
        # CORRECT WAY (no data leakage)
        print("\n✅ CORRECT WAY - No Data Leakage:")
        print("Fitting scaler only on training data")
        scaler_correct = StandardScaler()
        X_train_scaled = scaler_correct.fit_transform(X_train)  # CORRECT - only train data
        X_test_scaled = scaler_correct.transform(X_test)  # CORRECT - transform test with train parameters
        print("This approach keeps train and test data separate!")
        
        return X_train_scaled, X_test_scaled
    
    # Run the problems
    print("1. Building Preprocessing Pipeline:")
    preprocessor, X_preprocessed = build_preprocessing_pipeline()
    
    print("\n2. Handling Data Leakage:")
    X_train_scaled, X_test_scaled = handle_data_leakage()

# Run all problems
if __name__ == "__main__":
    print("=== Data Preprocessing Practice Problems ===\n")
    
    problem_1()
    print("\n" + "="*50 + "\n")
    
    problem_2()
    print("\n" + "="*50 + "\n")
    
    problem_3()
    print("\n" + "="*50 + "\n")
    
    problem_4()
    print("\n" + "="*50 + "\n")
    
    problem_5()
    print("\n" + "="*50 + "\n")
    
    print("=== Summary ===")
    print("These practice problems covered:")
    print("1. Data cleaning techniques for missing values and outliers")
    print("2. Feature scaling methods and when to use them")
    print("3. Categorical encoding strategies")
    print("4. Feature engineering approaches")
    print("5. Pipeline development and data leakage prevention")
    print("\nEach problem demonstrates:")
    print("- Implementation of specific preprocessing techniques")
    print("- Common problem patterns and solutions")
    print("- Best practices for real-world applications")
    print("- Potential pitfalls to avoid")