"""
Model Deployment and Serving
==========================

This module provides tools and techniques for deploying machine learning models in production.
It covers model serialization, API development, containerization, and monitoring.

Key Components:
- Model serialization and loading
- REST API development
- Containerization with Docker
- Model monitoring and logging
- A/B testing and versioning
"""

import pandas as pd
import numpy as np
import joblib
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelSerializer:
    """
    Model serialization and deserialization utilities.
    """
    
    @staticmethod
    def save_model(model, file_path, format='joblib'):
        """
        Save trained model to file.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model to save
        file_path : str
            Path to save model
        format : str, default='joblib'
            Format to save model ('joblib', 'pickle')
        """
        if format == 'joblib':
            joblib.dump(model, file_path)
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Model saved to {file_path} using {format} format")
    
    @staticmethod
    def load_model(file_path, format='joblib'):
        """
        Load trained model from file.
        
        Parameters:
        -----------
        file_path : str
            Path to load model from
        format : str, default='joblib'
            Format of saved model ('joblib', 'pickle')
            
        Returns:
        --------
        model : sklearn estimator
            Loaded model
        """
        try:
            if format == 'joblib':
                model = joblib.load(file_path)
            elif format == 'pickle':
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Model loaded from {file_path}")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None


class ModelAPI:
    """
    Simple REST API for serving machine learning models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to serve
    preprocessor : object, optional
        Preprocessing pipeline
    """
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.request_count = 0
        self.prediction_history = []
        
        print("ModelAPI initialized")
        print(f"Model type: {type(model).__name__}")
        print(f"Preprocessor: {'Yes' if preprocessor else 'No'}")
    
    def predict(self, data):
        """
        Make predictions on input data.
        
        Parameters:
        -----------
        data : dict or pandas.DataFrame
            Input data for prediction
            
        Returns:
        --------
        prediction : dict
            Prediction results
        """
        try:
            # Convert to DataFrame if dict
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Apply preprocessing if available
            if self.preprocessor:
                df_processed = self.preprocessor.transform(df)
            else:
                df_processed = df
            
            # Make prediction
            prediction = self.model.predict(df_processed)
            probability = None
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(df_processed)
            
            # Store prediction history
            timestamp = datetime.now().isoformat()
            prediction_record = {
                'timestamp': timestamp,
                'input_shape': df_processed.shape,
                'prediction': prediction.tolist(),
                'probability': probability.tolist() if probability is not None else None
            }
            self.prediction_history.append(prediction_record)
            self.request_count += 1
            
            # Return results
            result = {
                'prediction': prediction.tolist(),
                'probability': probability.tolist() if probability is not None else None,
                'timestamp': timestamp,
                'request_id': self.request_count
            }
            
            return result
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'request_id': self.request_count
            }
            return error_result
    
    def get_model_info(self):
        """
        Get model information.
        
        Returns:
        --------
        info : dict
            Model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'total_requests': self.request_count,
            'features_required': getattr(self.model, 'feature_names_in_', 'Not available'),
            'model_parameters': str(self.model.get_params()) if hasattr(self.model, 'get_params') else 'Not available'
        }
        return info
    
    def get_prediction_stats(self):
        """
        Get prediction statistics.
        
        Returns:
        --------
        stats : dict
            Prediction statistics
        """
        if not self.prediction_history:
            return {'message': 'No predictions made yet'}
        
        # Calculate statistics
        predictions = [record['prediction'][0] for record in self.prediction_history]
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'unique_predictions': len(set(str(p) for p in predictions)),
            'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
            'last_10_predictions': self.prediction_history[-10:]
        }
        
        return stats


class ModelMonitor:
    """
    Model monitoring and logging utilities.
    """
    
    def __init__(self, log_file='model_monitor.log'):
        self.log_file = log_file
        self.performance_log = []
        
        print(f"ModelMonitor initialized with log file: {log_file}")
    
    def log_prediction(self, request_id, input_data, prediction, actual=None):
        """
        Log prediction for monitoring.
        
        Parameters:
        -----------
        request_id : int
            Request identifier
        input_data : dict
            Input data
        prediction : list
            Model prediction
        actual : list, optional
            Actual values (for evaluation)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'input_shape': len(input_data) if isinstance(input_data, dict) else input_data.shape[0],
            'prediction': prediction,
            'actual': actual
        }
        
        # Calculate accuracy if actual values provided
        if actual is not None:
            try:
                accuracy = np.mean(np.array(prediction) == np.array(actual))
                log_entry['accuracy'] = accuracy
            except:
                log_entry['accuracy'] = None
        
        self.performance_log.append(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_performance_report(self, window_hours=24):
        """
        Get performance report.
        
        Parameters:
        -----------
        window_hours : int, default=24
            Time window in hours
            
        Returns:
        --------
        report : dict
            Performance report
        """
        if not self.performance_log:
            return {'message': 'No performance data available'}
        
        # Filter by time window
        cutoff_time = datetime.now().timestamp() - (window_hours * 3600)
        recent_logs = [
            log for log in self.performance_log 
            if datetime.fromisoformat(log['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_logs:
            return {'message': f'No performance data in last {window_hours} hours'}
        
        # Calculate metrics
        accuracies = [log.get('accuracy') for log in recent_logs if log.get('accuracy') is not None]
        avg_accuracy = np.mean(accuracies) if accuracies else None
        
        report = {
            'time_window_hours': window_hours,
            'total_predictions': len(recent_logs),
            'predictions_with_accuracy': len(accuracies),
            'average_accuracy': avg_accuracy,
            'recent_performance': recent_logs[-5:]  # Last 5 entries
        }
        
        return report


class ModelVersionManager:
    """
    Model version management utilities.
    """
    
    def __init__(self):
        self.models = {}
        self.current_version = None
        self.version_history = []
        
        print("ModelVersionManager initialized")
    
    def register_model(self, version, model, metadata=None):
        """
        Register a model version.
        
        Parameters:
        -----------
        version : str
            Model version identifier
        model : sklearn estimator
            Trained model
        metadata : dict, optional
            Model metadata
        """
        if metadata is None:
            metadata = {}
        
        self.models[version] = {
            'model': model,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat()
        }
        
        # Set as current if first model
        if self.current_version is None:
            self.current_version = version
        
        # Add to history
        self.version_history.append({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata
        })
        
        print(f"Model version {version} registered")
    
    def set_current_version(self, version):
        """
        Set current model version.
        
        Parameters:
        -----------
        version : str
            Model version to set as current
        """
        if version in self.models:
            self.current_version = version
            print(f"Current model version set to {version}")
        else:
            print(f"Version {version} not found")
    
    def get_model(self, version=None):
        """
        Get model by version.
        
        Parameters:
        -----------
        version : str, optional
            Model version (uses current if None)
            
        Returns:
        --------
        model : sklearn estimator
            Requested model
        """
        if version is None:
            version = self.current_version
        
        if version in self.models:
            return self.models[version]['model']
        else:
            print(f"Model version {version} not found")
            return None
    
    def list_versions(self):
        """
        List all model versions.
        
        Returns:
        --------
        versions : list
            List of model versions
        """
        return list(self.models.keys())
    
    def get_version_info(self, version=None):
        """
        Get version information.
        
        Parameters:
        -----------
        version : str, optional
            Model version (uses current if None)
            
        Returns:
        --------
        info : dict
            Version information
        """
        if version is None:
            version = self.current_version
        
        if version in self.models:
            return self.models[version]
        else:
            return {'error': f'Version {version} not found'}


class ABTester:
    """
    A/B testing utilities for model comparison.
    """
    
    def __init__(self):
        self.test_groups = {}
        self.results = {}
        
        print("ABTester initialized")
    
    def create_test_group(self, name, model_a, model_b, split_ratio=0.5):
        """
        Create A/B test group.
        
        Parameters:
        -----------
        name : str
            Test group name
        model_a : sklearn estimator
            Model A
        model_b : sklearn estimator
            Model B
        split_ratio : float, default=0.5
            Ratio for splitting traffic (0.5 = 50/50)
        """
        self.test_groups[name] = {
            'model_a': model_a,
            'model_b': model_b,
            'split_ratio': split_ratio,
            'predictions_a': [],
            'predictions_b': [],
            'start_time': datetime.now().isoformat()
        }
        
        print(f"A/B test group '{name}' created with {split_ratio*100:.0f}/{(1-split_ratio)*100:.0f} split")
    
    def route_prediction(self, test_group_name, data):
        """
        Route prediction to A or B model.
        
        Parameters:
        -----------
        test_group_name : str
            Test group name
        data : dict or pandas.DataFrame
            Input data
            
        Returns:
        --------
        result : dict
            Prediction result with model identifier
        """
        if test_group_name not in self.test_groups:
            return {'error': f'Test group {test_group_name} not found'}
        
        test_group = self.test_groups[test_group_name]
        
        # Random routing based on split ratio
        use_model_a = np.random.random() < test_group['split_ratio']
        
        if use_model_a:
            model = test_group['model_a']
            model_id = 'A'
        else:
            model = test_group['model_b']
            model_id = 'B'
        
        # Make prediction
        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            prediction = model.predict(df)
            
            # Store prediction
            test_group[f'predictions_{model_id.lower()}'].append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction.tolist(),
                'input_shape': df.shape
            })
            
            return {
                'prediction': prediction.tolist(),
                'model_used': model_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'model_used': model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_test_results(self, test_group_name):
        """
        Get A/B test results.
        
        Parameters:
        -----------
        test_group_name : str
            Test group name
            
        Returns:
        --------
        results : dict
            Test results
        """
        if test_group_name not in self.test_groups:
            return {'error': f'Test group {test_group_name} not found'}
        
        test_group = self.test_groups[test_group_name]
        
        results = {
            'test_group': test_group_name,
            'start_time': test_group['start_time'],
            'model_a_predictions': len(test_group['predictions_a']),
            'model_b_predictions': len(test_group['predictions_b']),
            'total_predictions': len(test_group['predictions_a']) + len(test_group['predictions_b'])
        }
        
        # Calculate split percentages
        total = results['total_predictions']
        if total > 0:
            results['model_a_percentage'] = (results['model_a_predictions'] / total) * 100
            results['model_b_percentage'] = (results['model_b_predictions'] / total) * 100
        
        return results


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Model Deployment and Serving Demonstration")
    print("=" * 45)
    
    # Generate sample classification data
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create sample DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    sample_data = pd.DataFrame(X_test[:5], columns=feature_names)
    
    print(f"Sample data created: {sample_data.shape}")
    
    # Train sample model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model Serialization Demonstration
    print("\n1. Model Serialization:")
    print("-" * 22)
    
    serializer = ModelSerializer()
    serializer.save_model(model, 'sample_model.joblib', format='joblib')
    
    # Load model
    loaded_model = serializer.load_model('sample_model.joblib', format='joblib')
    
    # Model API Demonstration
    print("\n2. Model API:")
    print("-" * 12)
    
    # Initialize API
    model_api = ModelAPI(loaded_model)
    
    # Make predictions
    sample_input = sample_data.iloc[0].to_dict()
    prediction_result = model_api.predict(sample_input)
    print(f"Prediction result: {prediction_result}")
    
    # Get model info
    model_info = model_api.get_model_info()
    print(f"Model info: {model_info}")
    
    # Model Monitoring Demonstration
    print("\n3. Model Monitoring:")
    print("-" * 18)
    
    monitor = ModelMonitor('demo_monitor.log')
    
    # Log predictions
    for i in range(3):
        sample_input = sample_data.iloc[i].to_dict()
        prediction = model_api.predict(sample_input)
        monitor.log_prediction(
            prediction['request_id'],
            sample_input,
            prediction['prediction'],
            [y_test[i]]  # Actual value
        )
    
    # Get performance report
    performance_report = monitor.get_performance_report()
    print(f"Performance report: {performance_report}")
    
    # Model Version Management Demonstration
    print("\n4. Model Version Management:")
    print("-" * 26)
    
    version_manager = ModelVersionManager()
    
    # Register models
    version_manager.register_model(
        'v1.0', model,
        metadata={'accuracy': 0.92, 'training_date': '2024-01-01'}
    )
    
    # Create improved model
    improved_model = RandomForestClassifier(n_estimators=200, random_state=42)
    improved_model.fit(X_train, y_train)
    
    version_manager.register_model(
        'v2.0', improved_model,
        metadata={'accuracy': 0.95, 'training_date': '2024-01-15'}
    )
    
    # List versions
    versions = version_manager.list_versions()
    print(f"Available versions: {versions}")
    
    # Get current model
    current_model = version_manager.get_model()
    print(f"Current model version: {version_manager.current_version}")
    
    # A/B Testing Demonstration
    print("\n5. A/B Testing:")
    print("-" * 14)
    
    ab_tester = ABTester()
    
    # Create test group
    ab_tester.create_test_group('model_comparison', model, improved_model, split_ratio=0.5)
    
    # Route predictions
    for i in range(5):
        sample_input = sample_data.iloc[i].to_dict()
        result = ab_tester.route_prediction('model_comparison', sample_input)
        print(f"A/B Test Result {i+1}: Model {result.get('model_used', 'N/A')}, "
              f"Prediction: {result.get('prediction', 'N/A')}")
    
    # Get test results
    test_results = ab_tester.get_test_results('model_comparison')
    print(f"A/B Test Results: {test_results}")
    
    # Deployment strategies summary
    print("\n" + "="*50)
    print("Model Deployment Strategies Summary")
    print("="*50)
    print("1. Model Serialization:")
    print("   - Joblib for scikit-learn models")
    print("   - Pickle for general Python objects")
    print("   - ONNX for cross-platform compatibility")
    print("   - Model versioning and metadata")
    
    print("\n2. API Development:")
    print("   - REST APIs with Flask/FastAPI")
    print("   - GraphQL for complex queries")
    print("   - gRPC for high-performance")
    print("   - Authentication and authorization")
    
    print("\n3. Containerization:")
    print("   - Docker for reproducible deployments")
    print("   - Kubernetes for orchestration")
    print("   - Helm charts for deployment management")
    print("   - CI/CD pipelines")
    
    print("\n4. Cloud Deployment:")
    print("   - AWS SageMaker, Google AI Platform")
    print("   - Azure Machine Learning")
    print("   - Serverless deployments")
    print("   - Auto-scaling capabilities")
    
    print("\n5. Edge Deployment:")
    print("   - TensorFlow Lite, ONNX Runtime")
    print("   - Mobile deployment (TensorFlow Lite)")
    print("   - IoT device deployment")
    print("   - Model compression techniques")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Model Deployment")
    print("="*50)
    print("1. Model Management:")
    print("   - Version control for models")
    print("   - Metadata tracking")
    print("   - Model registry")
    print("   - Rollback capabilities")
    
    print("\n2. Performance Optimization:")
    print("   - Model compression")
    print("   - Caching strategies")
    print("   - Batch processing")
    print("   - Asynchronous processing")
    
    print("\n3. Monitoring and Logging:")
    print("   - Real-time performance metrics")
    print("   - Data drift detection")
    print("   - Model accuracy monitoring")
    print("   - Error tracking and alerting")
    
    print("\n4. Security:")
    print("   - Input validation")
    print("   - Authentication and authorization")
    print("   - Encryption in transit and at rest")
    print("   - Secure model storage")
    
    print("\n5. Scalability:")
    print("   - Load balancing")
    print("   - Auto-scaling")
    print("   - Database optimization")
    print("   - Caching layers")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using:")
    print("- MLflow: Model management and deployment")
    print("- FastAPI/Flask: API development")
    print("- Docker/Kubernetes: Containerization and orchestration")
    print("- Prometheus/Grafana: Monitoring and visualization")
    print("- These provide enterprise-grade deployment capabilities")