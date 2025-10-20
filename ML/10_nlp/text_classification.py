"""
Text Classification Implementation
================================

This module demonstrates text classification techniques including sentiment analysis,
spam detection, and topic classification using various ML approaches.

Key Concepts:
- Sentiment Analysis
- Text Classification Algorithms
- Feature Engineering for Text
- Model Evaluation for NLP
- Cross-validation for Text Data
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


class TextClassifier:
    """
    A comprehensive text classifier supporting multiple algorithms.
    
    Parameters:
    -----------
    model_type : str, default='naive_bayes'
        Type of classifier ('naive_bayes', 'logistic_regression', 'svm', 'random_forest')
    vectorizer_type : str, default='tfidf'
        Type of vectorizer ('tfidf', 'count')
    max_features : int, default=10000
        Maximum number of features
    ngram_range : tuple, default=(1, 1)
        N-gram range for feature extraction
    """
    
    def __init__(self, model_type='naive_bayes', vectorizer_type='tfidf', 
                 max_features=10000, ngram_range=(1, 1)):
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizer
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, 
                                            ngram_range=ngram_range,
                                            stop_words='english')
        else:
            from sklearn.feature_extraction.text import CountVectorizer
            self.vectorizer = CountVectorizer(max_features=max_features,
                                           ngram_range=ngram_range,
                                           stop_words='english')
        
        # Initialize model
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.is_trained = False
        self.classes = None
    
    def fit(self, X, y):
        """
        Train the text classifier.
        
        Parameters:
        -----------
        X : list or array
            Training texts
        y : list or array
            Training labels
            
        Returns:
        --------
        self : TextClassifier
            Trained classifier
        """
        print(f"Training {self.model_type} text classifier...")
        
        # Fit pipeline
        self.pipeline.fit(X, y)
        self.is_trained = True
        self.classes = self.pipeline.classes_
        
        print("Training completed")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : list or array
            Texts to classify
            
        Returns:
        --------
        predictions : array
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : list or array
            Texts to classify
            
        Returns:
        --------
        probabilities : array
            Predicted class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.pipeline.named_steps['classifier'], 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            raise ValueError(f"{self.model_type} does not support probability prediction")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier on test data.
        
        Parameters:
        -----------
        X_test : list or array
            Test texts
        y_test : list or array
            Test labels
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : list or array
            Training texts
        y : list or array
            Training labels
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        scores : array
            Cross-validation scores
        """
        if not self.is_trained:
            # Temporarily fit for cross-validation
            self.pipeline.fit(X, y)
        
        scores = cross_val_score(self.pipeline, X, y, cv=cv)
        return scores


class SentimentAnalyzer:
    """
    Specialized sentiment analyzer for text data.
    
    Parameters:
    -----------
    method : str, default='naive_bayes'
        Classification method
    """
    
    def __init__(self, method='naive_bayes'):
        self.method = method
        self.classifier = TextClassifier(model_type=method)
        self.is_trained = False
    
    def fit(self, texts, sentiments):
        """
        Train the sentiment analyzer.
        
        Parameters:
        -----------
        texts : list
            Training texts
        sentiments : list
            Training sentiment labels (e.g., 'positive', 'negative', 'neutral')
            
        Returns:
        --------
        self : SentimentAnalyzer
            Trained analyzer
        """
        print("Training sentiment analyzer...")
        self.classifier.fit(texts, sentiments)
        self.is_trained = True
        print("Sentiment analyzer training completed")
        return self
    
    def predict_sentiment(self, texts):
        """
        Predict sentiment of texts.
        
        Parameters:
        -----------
        texts : list
            Texts to analyze
            
        Returns:
        --------
        sentiments : list
            Predicted sentiments
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before prediction")
        
        return self.classifier.predict(texts)
    
    def predict_sentiment_proba(self, texts):
        """
        Predict sentiment probabilities.
        
        Parameters:
        -----------
        texts : list
            Texts to analyze
            
        Returns:
        --------
        probabilities : array
            Predicted sentiment probabilities
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before prediction")
        
        return self.classifier.predict_proba(texts)
    
    def get_sentiment_score(self, text):
        """
        Get sentiment score for a single text.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        score : float
            Sentiment score (positive values indicate positive sentiment)
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before prediction")
        
        # Simple scoring based on predicted class
        # In practice, you might use probability scores
        prediction = self.classifier.predict([text])[0]
        
        # Map sentiment to score
        sentiment_scores = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
        return sentiment_scores.get(prediction, 0.0)


class SpamDetector:
    """
    Email spam detection system.
    
    Parameters:
    -----------
    method : str, default='naive_bayes'
        Classification method
    """
    
    def __init__(self, method='naive_bayes'):
        self.method = method
        self.classifier = TextClassifier(model_type=method)
        self.is_trained = False
    
    def fit(self, emails, labels):
        """
        Train the spam detector.
        
        Parameters:
        -----------
        emails : list
            Training emails
        labels : list
            Training labels ('spam' or 'ham')
            
        Returns:
        --------
        self : SpamDetector
            Trained detector
        """
        print("Training spam detector...")
        self.classifier.fit(emails, labels)
        self.is_trained = True
        print("Spam detector training completed")
        return self
    
    def predict_spam(self, emails):
        """
        Predict if emails are spam.
        
        Parameters:
        -----------
        emails : list
            Emails to classify
            
        Returns:
        --------
        predictions : list
            Spam predictions
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        return self.classifier.predict(emails)
    
    def is_spam(self, email):
        """
        Check if a single email is spam.
        
        Parameters:
        -----------
        email : str
            Email to check
            
        Returns:
        --------
        is_spam : bool
            True if spam, False otherwise
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before prediction")
        
        prediction = self.classifier.predict([email])[0]
        return prediction == 'spam'


# Example usage and demonstration
if __name__ == "__main__":
    # Sample data for demonstration
    print("Text Classification Demonstration")
    print("=" * 50)
    
    # Sample sentiment data
    sentiment_texts = [
        "I love this movie, it's absolutely fantastic!",
        "This film is terrible, worst movie ever",
        "Great acting and wonderful story",
        "Boring plot and bad acting",
        "Amazing cinematography and direction",
        "Waste of time, very disappointing",
        "The movie was okay, nothing special",
        "Outstanding performance by all actors",
        "Poor script and direction",
        "Excellent film, highly recommended"
    ]
    
    sentiment_labels = [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'neutral', 'positive', 'negative', 'positive'
    ]
    
    # Sample spam data
    spam_texts = [
        "Congratulations! You've won $1000000! Click here to claim now!",
        "Hi John, let's meet for lunch tomorrow at 12pm",
        "URGENT: Your account will be closed! Verify now!",
        "Meeting notes from yesterday's conference call",
        "FREE VIAGRA! No prescription needed! Order now!",
        "Please review the quarterly report attached",
        "You have inherited $5000000 from a distant relative!",
        "Reminder: Team meeting scheduled for Friday 3pm",
        "CLICK HERE FOR AMAZING DEALS!!! LIMITED TIME OFFER!!!",
        "Happy birthday! Hope you have a wonderful day!"
    ]
    
    spam_labels = [
        'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham'
    ]
    
    # Sentiment Analysis
    print("\n1. Sentiment Analysis:")
    
    # Train sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer(method='naive_bayes')
    sentiment_analyzer.fit(sentiment_texts, sentiment_labels)
    
    # Test sentiment analysis
    test_texts = [
        "This movie is absolutely wonderful!",
        "I hate this film, it's boring",
        "The movie was okay, not great but not bad either"
    ]
    
    sentiments = sentiment_analyzer.predict_sentiment(test_texts)
    scores = [sentiment_analyzer.get_sentiment_score(text) for text in test_texts]
    
    for text, sentiment, score in zip(test_texts, sentiments, scores):
        print(f"Text: '{text}'")
        print(f"  Predicted sentiment: {sentiment}")
        print(f"  Sentiment score: {score:.2f}")
        print()
    
    # Spam Detection
    print("\n2. Spam Detection:")
    
    # Train spam detector
    spam_detector = SpamDetector(method='naive_bayes')
    spam_detector.fit(spam_texts, spam_labels)
    
    # Test spam detection
    test_emails = [
        "Congratulations! You've won a prize! Click here!",
        "Hi, can we schedule a meeting for next week?",
        "URGENT: Verify your account immediately!"
    ]
    
    spam_predictions = spam_detector.predict_spam(test_emails)
    
    for email, is_spam in zip(test_emails, spam_predictions):
        print(f"Email: '{email}'")
        print(f"  Spam prediction: {is_spam}")
        print()
    
    # Compare different classification methods
    print("\n" + "="*50)
    print("Comparison of Classification Methods")
    print("="*50)
    
    # Split sentiment data
    X_train, X_test, y_train, y_test = train_test_split(
        sentiment_texts, sentiment_labels, test_size=0.3, random_state=42)
    
    # Test different models
    models = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
    results = {}
    
    for model_type in models:
        try:
            classifier = TextClassifier(model_type=model_type)
            classifier.fit(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test)
            results[model_type] = metrics['accuracy']
            print(f"{model_type.capitalize()}: {metrics['accuracy']:.3f}")
        except Exception as e:
            print(f"{model_type.capitalize()}: Error - {str(e)}")
            results[model_type] = 0
    
    # Cross-validation comparison
    print("\nCross-validation scores (5-fold):")
    for model_type in models:
        try:
            classifier = TextClassifier(model_type=model_type)
            cv_scores = classifier.cross_validate(sentiment_texts, sentiment_labels, cv=3)
            print(f"{model_type.capitalize()}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except Exception as e:
            print(f"{model_type.capitalize()}: Error - {str(e)}")
    
    # Feature engineering demonstration
    print("\n" + "="*50)
    print("Feature Engineering Comparison")
    print("="*50)
    
    # Compare TF-IDF vs Count vectorization
    vectorizers = ['tfidf', 'count']
    for vect_type in vectorizers:
        try:
            classifier = TextClassifier(model_type='naive_bayes', vectorizer_type=vect_type)
            classifier.fit(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test)
            print(f"{vect_type.capitalize()} Vectorization: {metrics['accuracy']:.3f}")
        except Exception as e:
            print(f"{vect_type.capitalize()} Vectorization: Error - {str(e)}")
    
    # Compare n-gram ranges
    print("\nN-gram Range Comparison:")
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    for ngram in ngram_ranges:
        try:
            classifier = TextClassifier(model_type='naive_bayes', ngram_range=ngram)
            classifier.fit(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test)
            print(f"N-grams {ngram}: {metrics['accuracy']:.3f}")
        except Exception as e:
            print(f"N-grams {ngram}: Error - {str(e)}")
    
    # Advanced text classification concepts
    print("\n" + "="*50)
    print("Advanced Text Classification Concepts")
    print("="*50)
    print("1. Ensemble Methods:")
    print("   - Combine multiple classifiers for better performance")
    print("   - Voting classifiers, stacking, bagging")
    
    print("\n2. Deep Learning Approaches:")
    print("   - Recurrent Neural Networks for sequences")
    print("   - Convolutional Neural Networks for text")
    print("   - Transformer models (BERT, RoBERTa)")
    
    print("\n3. Transfer Learning:")
    print("   - Fine-tune pre-trained language models")
    print("   - Use embeddings from Word2Vec, GloVe, FastText")
    
    print("\n4. Active Learning:")
    print("   - Select most informative samples for labeling")
    print("   - Reduce annotation costs")
    
    # Evaluation metrics explanation
    print("\n" + "="*50)
    print("Important Evaluation Metrics for Text Classification")
    print("="*50)
    print("1. Accuracy:")
    print("   - Overall correct predictions")
    print("   - Good for balanced datasets")
    
    print("\n2. Precision and Recall:")
    print("   - Precision: TP / (TP + FP) - How many selected items are relevant?")
    print("   - Recall: TP / (TP + FN) - How many relevant items are selected?")
    
    print("\n3. F1-Score:")
    print("   - Harmonic mean of precision and recall")
    print("   - Good for imbalanced datasets")
    
    print("\n4. Confusion Matrix:")
    print("   - Shows true vs predicted classifications")
    print("   - Helps identify specific misclassification patterns")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Text Classification")
    print("="*50)
    print("1. Data Quality:")
    print("   - Ensure high-quality, representative training data")
    print("   - Handle class imbalance appropriately")
    
    print("\n2. Preprocessing:")
    print("   - Apply consistent text preprocessing")
    print("   - Consider domain-specific cleaning")
    
    print("\n3. Feature Engineering:")
    print("   - Experiment with different vectorization methods")
    print("   - Try various n-gram ranges")
    print("   - Consider feature selection techniques")
    
    print("\n4. Model Selection:")
    print("   - Compare multiple algorithms")
    print("   - Use cross-validation for robust evaluation")
    print("   - Consider ensemble methods")
    
    print("\n5. Evaluation:")
    print("   - Use appropriate metrics for your task")
    print("   - Validate on held-out test set")
    print("   - Monitor for overfitting")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- scikit-learn: Comprehensive ML toolkit")
    print("- spaCy: Industrial-strength NLP library")
    print("- transformers (Hugging Face): Pre-trained models")
    print("- These provide optimized implementations and pre-trained models")