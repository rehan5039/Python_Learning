"""
Machine Learning Project Framework
================================

This module provides a comprehensive framework for executing end-to-end machine learning projects.
It covers all phases from problem definition to deployment and monitoring.

Key Components:
- Project initialization and configuration
- Data loading and exploration
- Preprocessing pipeline
- Model training and evaluation
- Deployment utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class MLProject:
    """
    Comprehensive ML project framework.
    
    Parameters:
    -----------
    project_name : str
        Name of the project
    problem_type : str, default='classification'
        Type of ML problem ('classification', 'regression', 'clustering')
    """
    
    def __init__(self, project_name, problem_type='classification'):
        self.project_name = project_name
        self.problem_type = problem_type
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.metrics = {}
        self.feature_names = None
        
        print(f"ML Project '{project_name}' initialized")
        print(f"Problem type: {problem_type}")
    
    def load_data(self, file_path, target_column=None):
        """
        Load data from file.
        
        Parameters:
        -----------
        file_path : str
            Path to data file (CSV, Excel, etc.)
        target_column : str, optional
            Name of target column
            
        Returns:
        --------
        data : pandas.DataFrame
            Loaded data
        """
        try:
            # Load data based on file extension
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
            print(f"Data loaded successfully: {self.data.shape}")
            
            # If target column specified, separate features and target
            if target_column and target_column in self.data.columns:
                self.X = self.data.drop(columns=[target_column])
                self.y = self.data[target_column]
                self.feature_names = self.X.columns.tolist()
                print(f"Features: {len(self.feature_names)}")
                print(f"Target column: {target_column}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def explore_data(self):
        """
        Perform exploratory data analysis.
        
        Returns:
        --------
        info : dict
            Dictionary containing data exploration results
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("Data Exploration Results:")
        print("=" * 30)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print("\nData types:")
        print(self.data.dtypes.value_counts())
        
        # Missing values
        missing_values = self.data.isnull().sum()
        missing_percent = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        
        print("\nMissing values:")
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Basic statistics
        print("\nBasic statistics:")
        print(self.data.describe())
        
        # Target distribution (if available)
        if self.y is not None:
            print("\nTarget distribution:")
            if self.problem_type == 'classification':
                print(self.y.value_counts())
            else:
                print(self.y.describe())
        
        return {
            'shape': self.data.shape,
            'missing_values': missing_df,
            'data_types': self.data.dtypes.value_counts(),
            'basic_stats': self.data.describe()
        }
    
    def preprocess_data(self, handle_missing='mean', encode_categorical=True, 
                       scale_features=True, test_size=0.2, random_state=42):
        """
        Preprocess data for modeling.
        
        Parameters:
        -----------
        handle_missing : str, default='mean'
            How to handle missing values ('mean', 'median', 'drop')
        encode_categorical : bool, default=True
            Whether to encode categorical variables
        scale_features : bool, default=True
            Whether to scale numerical features
        test_size : float, default=0.2
            Proportion of data for testing
        random_state : int, default=42
            Random state for reproducibility
            
        Returns:
        --------
        splits : tuple
            (X_train, X_test, y_train, y_test) or (X_scaled, None, y, None)
        """
        if self.X is None or self.y is None:
            print("Features and target not defined. Please load data with target column.")
            return None
        
        # Handle missing values
        if handle_missing == 'drop':
            # Drop rows with missing values
            complete_data = pd.concat([self.X, self.y], axis=1).dropna()
            self.X = complete_data.iloc[:, :-1]
            self.y = complete_data.iloc[:, -1]
        elif handle_missing == 'mean':
            # Fill with mean for numerical columns
            self.X = self.X.fillna(self.X.mean())
        elif handle_missing == 'median':
            # Fill with median for numerical columns
            self.X = self.X.fillna(self.X.median())
        
        # Encode categorical variables
        if encode_categorical:
            categorical_columns = self.X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                self.X[col] = self.label_encoder.fit_transform(self.X[col].astype(str))
            
            # Encode target if it's categorical
            if self.problem_type == 'classification' and self.y.dtype == 'object':
                self.y = self.label_encoder.fit_transform(self.y.astype(str))
        
        # Scale features
        if scale_features:
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns
            )
        
        # Split data
        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state,
                stratify=self.y if self.problem_type == 'classification' else None
            )
            print(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            print(f"Data prepared: {self.X.shape}")
            return self.X, None, self.y, None
    
    def train_model(self, X_train, y_train, model=None, cv_folds=5):
        """
        Train machine learning model.
        
        Parameters:
        -----------
        X_train : pandas.DataFrame or numpy.array
            Training features
        y_train : pandas.Series or numpy.array
            Training target
        model : sklearn estimator, optional
            Model to train (default: RandomForest)
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        model : trained model
            Trained machine learning model
        """
        # Use default model if none provided
        if model is None:
            if self.problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.model = model
        
        # Train model
        self.model.fit(X_train, y_train)
        print(f"Model trained: {type(self.model).__name__}")
        
        # Cross-validation
        if cv_folds > 1:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate trained model.
        
        Parameters:
        -----------
        X_test : pandas.DataFrame or numpy.array
            Test features
        y_test : pandas.Series or numpy.array
            Test target
        X_train : pandas.DataFrame or numpy.array, optional
            Training features for comparison
        y_train : pandas.Series or numpy.array, optional
            Training target for comparison
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if self.model is None:
            print("No model trained. Please train a model first.")
            return None
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics based on problem type
        if self.problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            self.metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print("Model Evaluation Results:")
            print("=" * 25)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            # Training performance comparison
            if X_train is not None and y_train is not None:
                y_train_pred = self.model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                print(f"\nTraining Accuracy: {train_accuracy:.4f}")
                print(f"Test Accuracy:     {accuracy:.4f}")
                print(f"Difference:        {abs(train_accuracy - accuracy):.4f}")
        
        else:  # Regression
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            self.metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2
            }
            
            print("Model Evaluation Results:")
            print("=" * 25)
            print(f"MSE:  {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²:   {r2:.4f}")
        
        return self.metrics
    
    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from trained model.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to return
            
        Returns:
        --------
        importance_df : pandas.DataFrame
            DataFrame with feature importance
        """
        if self.model is None:
            print("No model trained. Please train a model first.")
            return None
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(importance))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"Top {top_n} Important Features:")
            print("=" * 30)
            print(importance_df.head(top_n))
            
            return importance_df.head(top_n)
        else:
            print(f"Model {type(self.model).__name__} does not support feature importance")
            return None
    
    def predict(self, X_new):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X_new : pandas.DataFrame or numpy.array
            New data for prediction
            
        Returns:
        --------
        predictions : numpy.array
            Model predictions
        """
        if self.model is None:
            raise ValueError("No model trained. Please train a model first.")
        
        # Preprocess new data (scale if needed)
        if hasattr(self.scaler, 'scale_'):  # Check if scaler was fitted
            X_new_scaled = self.scaler.transform(X_new)
        else:
            X_new_scaled = X_new
        
        # Make predictions
        predictions = self.model.predict(X_new_scaled)
        return predictions
    
    def save_model(self, file_path):
        """
        Save trained model to file.
        
        Parameters:
        -----------
        file_path : str
            Path to save model
        """
        import joblib
        
        if self.model is None:
            print("No model to save. Please train a model first.")
            return
        
        # Save model and preprocessing components
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'problem_type': self.problem_type
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        """
        Load trained model from file.
        
        Parameters:
        -----------
        file_path : str
            Path to load model from
        """
        import joblib
        
        try:
            model_data = joblib.load(file_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            self.problem_type = model_data['problem_type']
            print(f"Model loaded from {file_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Machine Learning Project Framework Demonstration")
    print("=" * 50)
    
    # Generate sample classification data
    from sklearn.datasets import make_classification
    X_sample, y_sample = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_classes=3, random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]
    sample_data = pd.DataFrame(X_sample, columns=feature_names)
    sample_data['target'] = y_sample
    
    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)
    print(f"Sample data created: {sample_data.shape}")
    
    # Initialize project
    project = MLProject("Sample Classification Project", problem_type='classification')
    
    # Load data
    project.load_data('sample_data.csv', target_column='target')
    
    # Explore data
    exploration_results = project.explore_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = project.preprocess_data()
    
    # Train model
    trained_model = project.train_model(X_train, y_train)
    
    # Evaluate model
    evaluation_metrics = project.evaluate_model(X_test, y_test, X_train, y_train)
    
    # Get feature importance
    importance = project.get_feature_importance(top_n=5)
    
    # Make predictions on new data
    new_data = X_test[:5]  # First 5 test samples
    predictions = project.predict(new_data)
    print(f"\nPredictions on new data: {predictions}")
    
    # Save model
    project.save_model('sample_model.pkl')
    
    # Load model
    new_project = MLProject("Loaded Model Project")
    new_project.load_model('sample_model.pkl')
    
    # Project framework summary
    print("\n" + "="*50)
    print("ML Project Framework Summary")
    print("="*50)
    print("1. Project Initialization:")
    print("   - Define project scope and objectives")
    print("   - Set up project structure and documentation")
    
    print("\n2. Data Loading and Exploration:")
    print("   - Load data from various sources")
    print("   - Perform comprehensive EDA")
    print("   - Identify data quality issues")
    
    print("\n3. Data Preprocessing:")
    print("   - Handle missing values")
    print("   - Encode categorical variables")
    print("   - Scale numerical features")
    print("   - Split data appropriately")
    
    print("\n4. Model Training:")
    print("   - Select appropriate algorithms")
    print("   - Train baseline models")
    print("   - Perform cross-validation")
    print("   - Tune hyperparameters")
    
    print("\n5. Model Evaluation:")
    print("   - Calculate relevant metrics")
    print("   - Compare training and test performance")
    print("   - Analyze feature importance")
    print("   - Validate model assumptions")
    
    print("\n6. Model Deployment:")
    print("   - Save trained models")
    print("   - Create prediction interfaces")
    print("   - Set up monitoring systems")
    print("   - Document deployment process")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for ML Projects")
    print("="*50)
    print("1. Reproducibility:")
    print("   - Set random seeds")
    print("   - Version control code and data")
    print("   - Document environments and dependencies")
    
    print("\n2. Data Management:")
    print("   - Validate data quality")
    print("   - Handle data drift")
    print("   - Implement data privacy measures")
    print("   - Create data pipelines")
    
    print("\n3. Model Development:")
    print("   - Start with simple baselines")
    print("   - Use appropriate validation strategies")
    print("   - Monitor for overfitting")
    print("   - Consider model interpretability")
    
    print("\n4. Evaluation and Monitoring:")
    print("   - Use domain-appropriate metrics")
    print("   - Test on diverse datasets")
    print("   - Monitor performance in production")
    print("   - Plan for model updates")
    
    print("\n5. Documentation:")
    print("   - Maintain project documentation")
    print("   - Document model decisions")
    print("   - Create user guides")
    print("   - Share results effectively")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using:")
    print("- MLflow: Experiment tracking and model management")
    print("- DVC: Data version control")
    print("- Docker: Containerization for deployment")
    print("- Kubernetes: Orchestration for scaling")
    print("- These provide enterprise-grade ML lifecycle management")