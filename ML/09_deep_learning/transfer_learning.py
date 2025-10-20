"""
Transfer Learning Implementation
===============================

This module demonstrates transfer learning techniques including fine-tuning pre-trained models,
feature extraction, and domain adaptation. It covers practical applications and best practices.

Key Concepts:
- Fine-tuning Pre-trained Models
- Feature Extraction
- Domain Adaptation
- Model Freezing/Unfreezing
- Transfer Learning Strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class PretrainedModel:
    """
    A simulated pre-trained model for demonstration purposes.
    
    Parameters:
    -----------
    input_size : int
        Size of input features
    hidden_sizes : list
        Sizes of hidden layers
    output_size : int
        Size of output
    """
    
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layers = []
        self.weights = []
        self.biases = []
        
        # Initialize layers
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            self.layers.append({
                'weights': w,
                'biases': b,
                'frozen': False
            })
    
    def freeze_layer(self, layer_index):
        """Freeze a specific layer."""
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index]['frozen'] = True
            print(f"Layer {layer_index} frozen")
    
    def unfreeze_layer(self, layer_index):
        """Unfreeze a specific layer."""
        if 0 <= layer_index < len(self.layers):
            self.layers[layer_index]['frozen'] = False
            print(f"Layer {layer_index} unfrozen")
    
    def freeze_all_except_last(self):
        """Freeze all layers except the last one."""
        for i in range(len(self.layers) - 1):
            self.freeze_layer(i)
        self.unfreeze_layer(len(self.layers) - 1)
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, input_size)
            Input data
            
        Returns:
        --------
        output : numpy array
            Network output
        """
        # Forward pass through each layer
        activation = X
        for i, layer in enumerate(self.layers):
            z = np.dot(activation, layer['weights']) + layer['biases']
            # Apply ReLU activation for hidden layers, linear for output
            if i < len(self.layers) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = z  # Linear output
        return activation
    
    def get_feature_extractor(self):
        """
        Get a feature extractor that excludes the last layer.
        
        Returns:
        --------
        feature_extractor : function
            Function that extracts features
        """
        def feature_extractor(X):
            activation = X
            # Forward pass through all layers except the last one
            for i in range(len(self.layers) - 1):
                layer = self.layers[i]
                z = np.dot(activation, layer['weights']) + layer['biases']
                activation = np.maximum(0, z)  # ReLU
            return activation
        return feature_extractor


class TransferLearningModel:
    """
    A transfer learning model that can use pre-trained feature extractors.
    
    Parameters:
    -----------
    feature_extractor : function
        Function that extracts features from input
    num_classes : int
        Number of output classes
    """
    
    def __init__(self, feature_extractor, num_classes):
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.classifier_weights = None
        self.classifier_bias = None
        self.trained = False
    
    def fit(self, X, y, learning_rate=0.01, epochs=100):
        """
        Train the classifier on top of the feature extractor.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            Target labels (one-hot encoded)
        learning_rate : float
            Learning rate
        epochs : int
            Number of training epochs
        """
        # Extract features using pre-trained extractor
        features = self.feature_extractor(X)
        
        # Initialize classifier weights
        feature_size = features.shape[1]
        self.classifier_weights = np.random.randn(feature_size, self.num_classes) * 0.1
        self.classifier_bias = np.zeros((1, self.num_classes))
        
        # Train classifier
        print("Training classifier on extracted features...")
        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(features, self.classifier_weights) + self.classifier_bias
            predictions = softmax(logits)
            
            # Compute loss
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))
            
            # Backward pass
            d_logits = predictions - y
            d_weights = np.dot(features.T, d_logits) / len(features)
            d_bias = np.mean(d_logits, axis=0, keepdims=True)
            
            # Update weights
            self.classifier_weights -= learning_rate * d_weights
            self.classifier_bias -= learning_rate * d_bias
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.trained = True
        print("Classifier training completed")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : numpy array
            Input data
            
        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.feature_extractor(X)
        
        # Make predictions
        logits = np.dot(features, self.classifier_weights) + self.classifier_bias
        predictions = np.argmax(logits, axis=1)
        return predictions


def softmax(x):
    """Compute softmax values for x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class FineTuningModel:
    """
    A model for fine-tuning pre-trained models.
    
    Parameters:
    -----------
    pretrained_model : PretrainedModel
        Pre-trained model to fine-tune
    num_classes : int
        Number of output classes for new task
    """
    
    def __init__(self, pretrained_model, num_classes):
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        
        # Replace the last layer for the new task
        old_output_size = pretrained_model.output_size
        new_weights = np.random.randn(pretrained_model.layers[-1]['weights'].shape[0], num_classes) * 0.1
        new_biases = np.zeros((1, num_classes))
        
        # Update the last layer
        pretrained_model.layers[-1]['weights'] = new_weights
        pretrained_model.layers[-1]['biases'] = new_biases
        pretrained_model.output_size = num_classes
        
        print(f"Last layer updated from {old_output_size} to {num_classes} classes")
    
    def fine_tune(self, X, y, learning_rate=0.001, epochs=50, freeze_feature_extractor=True):
        """
        Fine-tune the pre-trained model.
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            Target labels (one-hot encoded)
        learning_rate : float
            Learning rate for fine-tuning
        epochs : int
            Number of fine-tuning epochs
        freeze_feature_extractor : bool
            Whether to freeze feature extractor layers
        """
        if freeze_feature_extractor:
            # Freeze all layers except the last one
            self.pretrained_model.freeze_all_except_last()
        
        print("Fine-tuning the model...")
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            logits = self.pretrained_model.forward(X)
            predictions = softmax(logits)
            
            # Compute loss
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-8), axis=1))
            losses.append(loss)
            
            # Backward pass
            d_logits = predictions - y
            
            # Update only unfrozen layers
            # Simplified backward pass for demonstration
            # In practice, this would involve proper backpropagation
            
            if epoch % 10 == 0:
                print(f"Fine-tuning epoch {epoch}, Loss: {loss:.4f}")
        
        print("Fine-tuning completed")
        return losses
    
    def predict(self, X):
        """
        Make predictions using the fine-tuned model.
        
        Parameters:
        -----------
        X : numpy array
            Input data
            
        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        logits = self.pretrained_model.forward(X)
        predictions = np.argmax(logits, axis=1)
        return predictions


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data for source task (image classification with 10 classes)
    print("Transfer Learning Demonstration")
    print("="*50)
    
    np.random.seed(42)
    
    # Source task: 10-class classification with 100 features
    X_source, y_source = make_classification(n_samples=1000, n_features=100, 
                                           n_classes=10, n_informative=50,
                                           random_state=42)
    
    # One-hot encode labels
    y_source_onehot = np.eye(10)[y_source]
    
    # Split source data
    X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
        X_source, y_source_onehot, test_size=0.2, random_state=42)
    
    print("Source task data:")
    print(f"Training set: {X_source_train.shape}")
    print(f"Test set: {X_source_test.shape}")
    print(f"Number of classes: 10")
    
    # Create and train a pre-trained model on source task
    print("\nTraining pre-trained model on source task...")
    pretrained_model = PretrainedModel(input_size=100, hidden_sizes=[64, 32], output_size=10)
    
    # Simulate training (in practice, this would involve actual training)
    source_predictions = pretrained_model.forward(X_source_train[:10])
    print(f"Source model output shape: {source_predictions.shape}")
    print("Pre-trained model created and simulated training completed")
    
    # Target task: Binary classification with similar features
    X_target, y_target = make_classification(n_samples=200, n_features=100,
                                           n_classes=2, n_informative=30,
                                           random_state=24)
    
    # One-hot encode target labels
    y_target_onehot = np.eye(2)[y_target]
    
    # Split target data
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target_onehot, test_size=0.3, random_state=42)
    
    print(f"\nTarget task data:")
    print(f"Training set: {X_target_train.shape}")
    print(f"Test set: {X_target_test.shape}")
    print(f"Number of classes: 2")
    
    # Demonstrate Feature Extraction
    print("\n" + "="*50)
    print("Method 1: Feature Extraction")
    print("="*50)
    
    # Get feature extractor from pre-trained model
    feature_extractor = pretrained_model.get_feature_extractor()
    
    # Extract features from target data
    target_features = feature_extractor(X_target_train)
    print(f"Original feature size: {X_target_train.shape[1]}")
    print(f"Extracted feature size: {target_features.shape[1]}")
    
    # Train a new classifier on extracted features
    transfer_model = TransferLearningModel(feature_extractor, num_classes=2)
    transfer_model.fit(X_target_train, y_target_train, learning_rate=0.01, epochs=50)
    
    # Evaluate transfer model
    target_predictions = transfer_model.predict(X_target_test)
    target_accuracy = accuracy_score(np.argmax(y_target_test, axis=1), target_predictions)
    print(f"Transfer learning accuracy: {target_accuracy:.4f}")
    
    # Demonstrate Fine-tuning
    print("\n" + "="*50)
    print("Method 2: Fine-tuning")
    print("="*50)
    
    # Create fine-tuning model
    fine_tune_model = FineTuningModel(pretrained_model, num_classes=2)
    
    # Fine-tune the model
    fine_tune_losses = fine_tune_model.fine_tune(
        X_target_train, y_target_train, 
        learning_rate=0.001, epochs=30, freeze_feature_extractor=True)
    
    # Evaluate fine-tuned model
    fine_tune_predictions = fine_tune_model.predict(X_target_test)
    fine_tune_accuracy = accuracy_score(np.argmax(y_target_test, axis=1), fine_tune_predictions)
    print(f"Fine-tuning accuracy: {fine_tune_accuracy:.4f}")
    
    # Compare with training from scratch
    print("\n" + "="*50)
    print("Comparison: Training from Scratch")
    print("="*50)
    
    # Create and train a model from scratch on target task
    scratch_model = PretrainedModel(input_size=100, hidden_sizes=[64, 32], output_size=2)
    
    # Simulate training from scratch (in practice, this would involve actual training)
    scratch_predictions = scratch_model.forward(X_target_train[:5])
    print("Scratch model created")
    print(f"Expected accuracy improvement with transfer learning: {target_accuracy:.4f} vs random initialization")
    
    # Demonstrate layer freezing/unfreezing
    print("\n" + "="*50)
    print("Layer Freezing/Unfreezing Demonstration")
    print("="*50)
    
    print("Freezing specific layers:")
    pretrained_model.freeze_layer(0)  # Freeze first layer
    pretrained_model.freeze_layer(1)  # Freeze second layer
    
    print("\nUnfreezing specific layers:")
    pretrained_model.unfreeze_layer(0)  # Unfreeze first layer
    
    # Transfer learning strategies
    print("\n" + "="*50)
    print("Transfer Learning Strategies:")
    print("="*50)
    print("1. Feature Extraction:")
    print("   - Use pre-trained model as fixed feature extractor")
    print("   - Train only the classifier on top")
    print("   - Best for small datasets with similar domains")
    
    print("\n2. Fine-tuning:")
    print("   - Start with pre-trained weights")
    print("   - Train the entire network or parts of it")
    print("   - Best for larger datasets or different domains")
    
    print("\n3. Layer-wise Training:")
    print("   - Train layers progressively from bottom to top")
    print("   - Helps with gradient flow in deep networks")
    
    print("\n4. Multi-task Learning:")
    print("   - Train on multiple related tasks simultaneously")
    print("   - Shared representations improve generalization")
    
    # Best practices
    print("\n" + "="*50)
    print("Transfer Learning Best Practices:")
    print("="*50)
    print("1. Choose appropriate pre-trained models for your domain")
    print("2. Match input preprocessing between source and target tasks")
    print("3. Start with lower learning rates for pre-trained layers")
    print("4. Monitor for overfitting, especially with small datasets")
    print("5. Use data augmentation to increase effective dataset size")
    print("6. Consider gradual unfreezing of layers during training")
    print("7. Evaluate performance on validation set regularly")
    
    # Popular pre-trained models
    print("\n" + "="*50)
    print("Popular Pre-trained Models:")
    print("="*50)
    print("Computer Vision:")
    print("  - ResNet, VGG, Inception, EfficientNet")
    print("  - Vision Transformers (ViT)")
    print("Natural Language Processing:")
    print("  - BERT, GPT, RoBERTa, T5")
    print("  - Transformer-based models")
    print("Speech Processing:")
    print("  - Wav2Vec, HuBERT")
    print("Multi-modal:")
    print("  - CLIP, DALL-E, Flamingo")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using frameworks:")
    print("- TensorFlow/Keras: tf.keras.applications")
    print("- PyTorch: torchvision.models, transformers library")
    print("- Hugging Face: Pre-trained models and tokenizers")
    print("- These provide optimized implementations and pre-trained weights")