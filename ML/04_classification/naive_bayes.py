"""
Naive Bayes Classifier Implementation

This module covers the implementation of Naive Bayes algorithms:
- Gaussian Naive Bayes for continuous features
- Multinomial Naive Bayes for discrete features
- Bernoulli Naive Bayes for binary features
- Laplace smoothing for zero probability handling
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features.
    
    Assumes features follow a normal (Gaussian) distribution.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.
        
        Args:
            var_smoothing: Portion of the largest variance added to variances for calculation stability
        """
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_priors = {}
        self.feature_means = {}
        self.feature_vars = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Train the Gaussian Naive Bayes classifier.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors P(y)
        for class_label in self.classes:
            self.class_priors[class_label] = np.sum(y == class_label) / n_samples
        
        # Calculate feature means and variances for each class
        for class_label in self.classes:
            # Get samples for this class
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Calculate mean and variance for each feature
            self.feature_means[class_label] = np.mean(X_class, axis=0)
            self.feature_vars[class_label] = np.var(X_class, axis=0)
            
            # Add smoothing to variances
            epsilon = self.var_smoothing * np.max(self.feature_vars[class_label])
            self.feature_vars[class_label] += epsilon
        
        return self
    
    def _gaussian_probability(self, x: float, mean: float, var: float) -> float:
        """
        Calculate Gaussian probability density.
        
        Args:
            x: Feature value
            mean: Mean of the distribution
            var: Variance of the distribution
            
        Returns:
            Probability density
        """
        # Avoid division by zero
        if var == 0:
            var = 1e-6
        
        # Calculate Gaussian probability density
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return coeff * exponent
    
    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample (n_features,)
            
        Returns:
            Predicted class label
        """
        class_probabilities = {}
        
        # Calculate posterior probability for each class
        for class_label in self.classes:
            # Start with class prior
            log_prob = np.log(self.class_priors[class_label])
            
            # Add log likelihood for each feature
            for i in range(len(x)):
                mean = self.feature_means[class_label][i]
                var = self.feature_vars[class_label][i]
                prob = self._gaussian_probability(x[i], mean, var)
                
                # Avoid log(0)
                if prob > 0:
                    log_prob += np.log(prob)
                else:
                    log_prob += np.log(1e-10)  # Small probability
            
            class_probabilities[class_label] = log_prob
        
        # Return class with highest posterior probability
        return max(class_probabilities, key=class_probabilities.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple samples.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Samples (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        probabilities = []
        
        for x in X:
            class_probs = {}
            
            # Calculate unnormalized log probabilities
            for class_label in self.classes:
                log_prob = np.log(self.class_priors[class_label])
                
                for i in range(len(x)):
                    mean = self.feature_means[class_label][i]
                    var = self.feature_vars[class_label][i]
                    prob = self._gaussian_probability(x[i], mean, var)
                    
                    if prob > 0:
                        log_prob += np.log(prob)
                    else:
                        log_prob += np.log(1e-10)
                
                class_probs[class_label] = log_prob
            
            # Normalize probabilities
            max_log_prob = max(class_probs.values())
            exp_probs = {k: np.exp(v - max_log_prob) for k, v in class_probs.items()}
            total_prob = sum(exp_probs.values())
            normalized_probs = {k: v / total_prob for k, v in exp_probs.items()}
            
            # Convert to array in class order
            prob_array = [normalized_probs[class_label] for class_label in self.classes]
            probabilities.append(prob_array)
        
        return np.array(probabilities)


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier for discrete features.
    
    Suitable for features representing counts or frequencies.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes classifier.
        
        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        self.feature_counts = {}
        self.class_counts = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Train the Multinomial Naive Bayes classifier.
        
        Args:
            X: Training features (n_samples, n_features) - non-negative integers
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        for class_label in self.classes:
            self.class_priors[class_label] = np.sum(y == class_label) / n_samples
        
        # Calculate feature counts and probabilities for each class
        for class_label in self.classes:
            # Get samples for this class
            class_mask = (y == class_label)
            X_class = X[class_mask]
            
            # Calculate total count for this class
            self.class_counts[class_label] = np.sum(X_class)
            
            # Calculate feature counts
            feature_counts = np.sum(X_class, axis=0)
            self.feature_counts[class_label] = feature_counts
            
            # Calculate feature probabilities with smoothing
            # P(feature_i | class) = (count(feature_i, class) + alpha) / (total_count(class) + alpha * n_features)
            self.feature_probs[class_label] = (feature_counts + self.alpha) / \
                                             (self.class_counts[class_label] + self.alpha * n_features)
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample (n_features,) - non-negative integers
            
        Returns:
            Predicted class label
        """
        class_scores = {}
        
        # Calculate log score for each class
        for class_label in self.classes:
            # Start with class prior
            log_score = np.log(self.class_priors[class_label])
            
            # Add log likelihood for each feature
            for i in range(len(x)):
                if x[i] > 0:
                    # For multinomial distribution: P(x_i | class)^(x_i)
                    log_score += x[i] * np.log(self.feature_probs[class_label][i])
            
            class_scores[class_label] = log_score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple samples.
        
        Args:
            X: Samples (n_samples, n_features) - non-negative integers
            
        Returns:
            Predicted class labels (n_samples,)
        """
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes classifier for binary features.
    
    Suitable for binary/boolean features.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Bernoulli Naive Bayes classifier.
        
        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BernoulliNaiveBayes':
        """
        Train the Bernoulli Naive Bayes classifier.
        
        Args:
            X: Training features (n_samples, n_features) - binary values (0 or 1)
            y: Training labels (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        for class_label in self.classes:
            self.class_priors[class_label] = np.sum(y == class_label) / n_samples
        
        # Calculate feature probabilities for each class
        for class_label in self.classes:
            # Get samples for this class
            class_mask = (y == class_label)
            X_class = X[class_mask]
            n_class_samples = X_class.shape[0]
            
            # Calculate probability of feature being 1 for this class
            # With Laplace smoothing: (count + alpha) / (n_class_samples + 2*alpha)
            feature_prob_1 = (np.sum(X_class, axis=0) + self.alpha) / \
                            (n_class_samples + 2 * self.alpha)
            
            self.feature_probs[class_label] = feature_prob_1
        
        return self
    
    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict class for a single sample.
        
        Args:
            x: Single sample (n_features,) - binary values (0 or 1)
            
        Returns:
            Predicted class label
        """
        class_scores = {}
        
        # Calculate log score for each class
        for class_label in self.classes:
            # Start with class prior
            log_score = np.log(self.class_priors[class_label])
            
            # Add log likelihood for each feature
            for i in range(len(x)):
                feature_prob = self.feature_probs[class_label][i]
                
                if x[i] == 1:
                    log_score += np.log(feature_prob)
                else:
                    log_score += np.log(1 - feature_prob)
            
            class_scores[class_label] = log_score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for multiple samples.
        
        Args:
            X: Samples (n_samples, n_features) - binary values (0 or 1)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)


def generate_continuous_data(n_samples: int = 1000, n_features: int = 5, 
                           n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate continuous data for Gaussian Naive Bayes.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y) features and labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features, 3),
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


def generate_discrete_data(n_samples: int = 1000, n_features: int = 100,
                          n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate discrete data for Multinomial Naive Bayes.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y) features and labels
    """
    # Generate random count data
    np.random.seed(42)
    X = np.random.poisson(lam=2, size=(n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def generate_binary_data(n_samples: int = 1000, n_features: int = 20,
                        n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary data for Bernoulli Naive Bayes.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y) features and labels
    """
    # Generate random binary data
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def compare_naive_bayes_variants(X_train_cont: np.ndarray, y_train_cont: np.ndarray,
                                X_test_cont: np.ndarray, y_test_cont: np.ndarray,
                                X_train_disc: np.ndarray, y_train_disc: np.ndarray,
                                X_test_disc: np.ndarray, y_test_disc: np.ndarray,
                                X_train_bin: np.ndarray, y_train_bin: np.ndarray,
                                X_test_bin: np.ndarray, y_test_bin: np.ndarray):
    """
    Compare different Naive Bayes variants.
    
    Args:
        X_train_cont, y_train_cont: Continuous training data
        X_test_cont, y_test_cont: Continuous test data
        X_train_disc, y_train_disc: Discrete training data
        X_test_disc, y_test_disc: Discrete test data
        X_train_bin, y_train_bin: Binary training data
        X_test_bin, y_test_bin: Binary test data
    """
    print("Naive Bayes Variants Comparison:")
    
    # Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_cont, y_train_cont)
    y_pred_gnb = gnb.predict(X_test_cont)
    accuracy_gnb = accuracy_score(y_test_cont, y_pred_gnb)
    print(f"  Gaussian Naive Bayes Accuracy: {accuracy_gnb:.4f}")
    
    # Multinomial Naive Bayes
    mnb = MultinomialNaiveBayes()
    mnb.fit(X_train_disc, y_train_disc)
    y_pred_mnb = mnb.predict(X_test_disc)
    accuracy_mnb = accuracy_score(y_test_disc, y_pred_mnb)
    print(f"  Multinomial Naive Bayes Accuracy: {accuracy_mnb:.4f}")
    
    # Bernoulli Naive Bayes
    bnb = BernoulliNaiveBayes()
    bnb.fit(X_train_bin, y_train_bin)
    y_pred_bnb = bnb.predict(X_test_bin)
    accuracy_bnb = accuracy_score(y_test_bin, y_pred_bnb)
    print(f"  Bernoulli Naive Bayes Accuracy: {accuracy_bnb:.4f}")


def demo():
    """Demonstrate Naive Bayes implementations."""
    print("=== Naive Bayes Classifier Demo ===\n")
    
    # Generate continuous data for Gaussian NB
    print("1. Gaussian Naive Bayes (Continuous Data):")
    X_cont, y_cont = generate_continuous_data(n_samples=1000, n_features=5, n_classes=3)
    X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(
        X_cont, y_cont, test_size=0.2, random_state=42
    )
    
    # Scale continuous data
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train_cont)
    X_test_cont_scaled = scaler.transform(X_test_cont)
    
    # Train Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_cont_scaled, y_train_cont)
    y_pred_gnb = gnb.predict(X_test_cont_scaled)
    accuracy_gnb = accuracy_score(y_test_cont, y_pred_gnb)
    
    print(f"   Accuracy: {accuracy_gnb:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_cont, y_pred_gnb))
    
    # Generate discrete data for Multinomial NB
    print("\n2. Multinomial Naive Bayes (Discrete Data):")
    X_disc, y_disc = generate_discrete_data(n_samples=1000, n_features=50, n_classes=3)
    X_train_disc, X_test_disc, y_train_disc, y_test_disc = train_test_split(
        X_disc, y_disc, test_size=0.2, random_state=42
    )
    
    # Train Multinomial Naive Bayes
    mnb = MultinomialNaiveBayes()
    mnb.fit(X_train_disc, y_train_disc)
    y_pred_mnb = mnb.predict(X_test_disc)
    accuracy_mnb = accuracy_score(y_test_disc, y_pred_mnb)
    
    print(f"   Accuracy: {accuracy_mnb:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_disc, y_pred_mnb))
    
    # Generate binary data for Bernoulli NB
    print("\n3. Bernoulli Naive Bayes (Binary Data):")
    X_bin, y_bin = generate_binary_data(n_samples=1000, n_features=20, n_classes=2)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42
    )
    
    # Train Bernoulli Naive Bayes
    bnb = BernoulliNaiveBayes()
    bnb.fit(X_train_bin, y_train_bin)
    y_pred_bnb = bnb.predict(X_test_bin)
    accuracy_bnb = accuracy_score(y_test_bin, y_pred_bnb)
    
    print(f"   Accuracy: {accuracy_bnb:.4f}")
    print(f"   Classification Report:")
    print(classification_report(y_test_bin, y_pred_bnb))
    
    # Compare with sklearn
    print("\n4. Comparison with Scikit-learn:")
    
    # Gaussian NB
    from sklearn.naive_bayes import GaussianNB
    sklearn_gnb = GaussianNB()
    sklearn_gnb.fit(X_train_cont_scaled, y_train_cont)
    sklearn_pred_gnb = sklearn_gnb.predict(X_test_cont_scaled)
    sklearn_accuracy_gnb = accuracy_score(y_test_cont, sklearn_pred_gnb)
    
    # Multinomial NB
    from sklearn.naive_bayes import MultinomialNB
    sklearn_mnb = MultinomialNB()
    sklearn_mnb.fit(X_train_disc, y_train_disc)
    sklearn_pred_mnb = sklearn_mnb.predict(X_test_disc)
    sklearn_accuracy_mnb = accuracy_score(y_test_disc, sklearn_pred_mnb)
    
    # Bernoulli NB
    from sklearn.naive_bayes import BernoulliNB
    sklearn_bnb = BernoulliNB()
    sklearn_bnb.fit(X_train_bin, y_train_bin)
    sklearn_pred_bnb = sklearn_bnb.predict(X_test_bin)
    sklearn_accuracy_bnb = accuracy_score(y_test_bin, sklearn_pred_bnb)
    
    print("   Our Implementation vs Scikit-learn:")
    print(f"   Gaussian NB: {accuracy_gnb:.4f} vs {sklearn_accuracy_gnb:.4f}")
    print(f"   Multinomial NB: {accuracy_mnb:.4f} vs {sklearn_accuracy_mnb:.4f}")
    print(f"   Bernoulli NB: {accuracy_bnb:.4f} vs {sklearn_accuracy_bnb:.4f}")


if __name__ == "__main__":
    demo()