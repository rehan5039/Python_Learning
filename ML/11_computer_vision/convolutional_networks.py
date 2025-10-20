"""
Convolutional Neural Networks for Computer Vision
==============================================

This module demonstrates Convolutional Neural Networks (CNNs) for computer vision tasks.
It covers CNN architectures, training techniques, and applications for image data.

Key Concepts:
- Convolutional Layers
- Pooling Layers
- CNN Architectures
- Transfer Learning
- Data Augmentation
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class Conv2D:
    """
    2D Convolutional Layer implementation.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (filters)
    kernel_size : int or tuple
        Size of convolutional kernel
    stride : int, default=1
        Stride of convolution
    padding : int, default=0
        Padding size
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        # Initialize weights and biases
        self.weights = np.random.randn(out_channels, in_channels, 
                                     self.kernel_size[0], self.kernel_size[1]) * 0.1
        self.biases = np.zeros((out_channels, 1))
        
        # Store for backward pass
        self.input = None
        self.output = None
    
    def forward(self, input_tensor):
        """
        Forward pass through convolutional layer.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        output : numpy array
            Output tensor after convolution
        """
        self.input = input_tensor
        batch_size, _, input_height, input_width = input_tensor.shape
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(input_tensor, 
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant', constant_values=0)
        else:
            padded_input = input_tensor
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Calculate input region
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract input region
                        input_region = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Compute convolution
                        output[b, oc, i, j] = np.sum(input_region * self.weights[oc]) + self.biases[oc]
        
        self.output = output
        return output


class MaxPool2D:
    """
    2D Max Pooling Layer implementation.
    
    Parameters:
    -----------
    pool_size : int or tuple
        Size of pooling window
    stride : int, optional
        Stride of pooling (defaults to pool_size)
    """
    
    def __init__(self, pool_size, stride=None):
        # Handle pool size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
        
        # Handle stride
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        
        # Store for backward pass
        self.input = None
        self.output = None
        self.max_indices = None
    
    def forward(self, input_tensor):
        """
        Forward pass through max pooling layer.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        output : numpy array
            Output tensor after pooling
        """
        self.input = input_tensor
        batch_size, channels, input_height, input_width = input_tensor.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size[0]) // self.stride[0] + 1
        output_width = (input_width - self.pool_size[1]) // self.stride[1] + 1
        
        # Initialize output and indices
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Calculate input region
                        h_start = i * self.stride[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.stride[1]
                        w_end = w_start + self.pool_size[1]
                        
                        # Extract input region
                        input_region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        
                        # Find maximum value and its indices
                        max_val = np.max(input_region)
                        max_idx = np.unravel_index(np.argmax(input_region), input_region.shape)
                        
                        # Store output and indices
                        output[b, c, i, j] = max_val
                        self.max_indices[b, c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        self.output = output
        return output


class ReLU:
    """
    Rectified Linear Unit activation function.
    """
    
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_tensor):
        """
        Forward pass through ReLU activation.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor
            
        Returns:
        --------
        output : numpy array
            Output tensor after ReLU activation
        """
        self.input = input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output


class Flatten:
    """
    Flatten layer to convert 4D tensor to 2D.
    """
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, input_tensor):
        """
        Forward pass through flatten layer.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        output : numpy array
            Flattened output tensor of shape (batch_size, features)
        """
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        return input_tensor.reshape(batch_size, -1)


class Dense:
    """
    Fully connected (Dense) layer.
    
    Parameters:
    -----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    """
    
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.biases = np.zeros((1, out_features))
        
        # Store for backward pass
        self.input = None
        self.output = None
    
    def forward(self, input_tensor):
        """
        Forward pass through dense layer.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor of shape (batch_size, in_features)
            
        Returns:
        --------
        output : numpy array
            Output tensor of shape (batch_size, out_features)
        """
        self.input = input_tensor
        self.output = np.dot(input_tensor, self.weights) + self.biases
        return self.output


class SimpleCNN:
    """
    Simple CNN implementation for educational purposes.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input images (channels, height, width)
    num_classes : int
        Number of output classes
    """
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Build network layers
        self.layers = []
        
        # Convolutional layers
        self.layers.append(Conv2D(input_shape[0], 32, 3, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(2, 2))
        
        self.layers.append(Conv2D(32, 64, 3, padding=1))
        self.layers.append(ReLU())
        self.layers.append(MaxPool2D(2, 2))
        
        # Flatten and fully connected layers
        self.layers.append(Flatten())
        self.layers.append(Dense(64 * (input_shape[1]//4) * (input_shape[2]//4), 128))
        self.layers.append(ReLU())
        self.layers.append(Dense(128, num_classes))
    
    def forward(self, input_tensor):
        """
        Forward pass through the entire network.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
        --------
        output : numpy array
            Output tensor of shape (batch_size, num_classes)
        """
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def predict(self, input_tensor):
        """
        Make predictions using the network.
        
        Parameters:
        -----------
        input_tensor : numpy array
            Input tensor
            
        Returns:
        --------
        predictions : numpy array
            Predicted class labels
        """
        logits = self.forward(input_tensor)
        return np.argmax(logits, axis=1)


class CNNTrainer:
    """
    Simple CNN trainer for educational purposes.
    
    Parameters:
    -----------
    model : SimpleCNN
        CNN model to train
    learning_rate : float, default=0.001
        Learning rate for optimization
    """
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.learning_rate = learning_rate
    
    def softmax(self, x):
        """
        Softmax activation function.
        
        Parameters:
        -----------
        x : numpy array
            Input array
            
        Returns:
        --------
        output : numpy array
            Softmax output
        """
        # Subtract max for numerical stability
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Parameters:
        -----------
        predictions : numpy array
            Predicted probabilities
        targets : numpy array
            True labels (one-hot encoded)
            
        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        predictions = np.clip(predictions, 1e-8, 1 - 1e-8)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))
    
    def train_step(self, X_batch, y_batch):
        """
        Perform one training step.
        
        Parameters:
        -----------
        X_batch : numpy array
            Input batch
        y_batch : numpy array
            Target batch (one-hot encoded)
            
        Returns:
        --------
        loss : float
            Training loss for the batch
        """
        # Forward pass
        logits = self.model.forward(X_batch)
        predictions = self.softmax(logits)
        
        # Compute loss
        loss = self.cross_entropy_loss(predictions, y_batch)
        
        # In a real implementation, we would compute gradients and update weights
        # For this educational example, we'll just return the loss
        return loss
    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the CNN model.
        
        Parameters:
        -----------
        X_train : numpy array
            Training data
        y_train : numpy array
            Training labels (one-hot encoded)
        epochs : int, default=10
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
            
        Returns:
        --------
        losses : list
            Training losses for each epoch
        """
        print("Training CNN model...")
        losses = []
        
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Process batches
            for i in range(0, num_samples, batch_size):
                # Get batch
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # Training step
                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Training completed")
        return losses


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Convolutional Neural Networks for Computer Vision Demonstration")
    print("=" * 65)
    
    # Create sample images (3 channels, 32x32)
    batch_size = 4
    input_shape = (3, 32, 32)
    sample_images = np.random.randn(batch_size, *input_shape).astype(np.float32)
    
    print(f"Sample batch shape: {sample_images.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Batch size: {batch_size}")
    
    # Conv2D Layer demonstration
    print("\n1. Conv2D Layer:")
    conv_layer = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    conv_output = conv_layer.forward(sample_images)
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {conv_output.shape}")
    print(f"Weight shape: {conv_layer.weights.shape}")
    
    # MaxPool2D Layer demonstration
    print("\n2. MaxPool2D Layer:")
    pool_layer = MaxPool2D(pool_size=2, stride=2)
    pool_output = pool_layer.forward(conv_output)
    print(f"Input shape: {conv_output.shape}")
    print(f"Output shape: {pool_output.shape}")
    
    # ReLU Layer demonstration
    print("\n3. ReLU Layer:")
    relu_layer = ReLU()
    relu_output = relu_layer.forward(pool_output)
    print(f"Input shape: {pool_output.shape}")
    print(f"Output shape: {relu_output.shape}")
    print(f"Non-zero elements: {np.count_nonzero(relu_output)}")
    
    # Simple CNN demonstration
    print("\n4. Simple CNN Model:")
    num_classes = 10
    cnn_model = SimpleCNN(input_shape, num_classes)
    
    # Forward pass
    cnn_output = cnn_model.forward(sample_images)
    print(f"CNN input shape: {sample_images.shape}")
    print(f"CNN output shape: {cnn_output.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Predictions
    predictions = cnn_model.predict(sample_images)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions}")
    
    # CNN Training demonstration
    print("\n5. CNN Training:")
    
    # Create sample training data
    X_train = np.random.randn(100, 3, 32, 32).astype(np.float32)
    y_train_onehot = np.eye(10)[np.random.randint(0, 10, 100)]
    
    # Create trainer
    trainer = CNNTrainer(cnn_model, learning_rate=0.001)
    
    # Train model (simplified training for demonstration)
    losses = trainer.train(X_train, y_train_onehot, epochs=5, batch_size=16)
    print(f"Training losses: {[f'{loss:.4f}' for loss in losses]}")
    
    # Compare different architectures
    print("\n" + "="*50)
    print("Comparison of CNN Architectures")
    print("="*50)
    
    # Performance comparison would typically involve:
    # 1. Different network depths
    # 2. Various filter sizes
    # 3. Different pooling strategies
    # 4. Residual connections
    # 5. Batch normalization
    
    print("Common CNN Architectures:")
    print("1. LeNet-5:")
    print("   - Simple architecture for digit recognition")
    print("   - 2 convolutional layers")
    print("   - 2 pooling layers")
    
    print("\n2. AlexNet:")
    print("   - 5 convolutional layers")
    print("   - 3 fully connected layers")
    print("   - Introduced ReLU activation")
    
    print("\n3. VGGNet:")
    print("   - Uniform 3x3 filters")
    print("   - Deep architecture (16-19 layers)")
    print("   - Small filters with deep networks")
    
    print("\n4. ResNet:")
    print("   - Residual connections")
    print("   - Very deep networks (50-152 layers)")
    print("   - Addresses vanishing gradient problem")
    
    print("\n5. Inception/GoogLeNet:")
    print("   - Multi-scale feature extraction")
    print("   - Inception modules")
    print("   - Efficient computation")
    
    # Advanced CNN concepts
    print("\n" + "="*50)
    print("Advanced CNN Concepts")
    print("="*50)
    print("1. Transfer Learning:")
    print("   - Fine-tuning pre-trained models")
    print("   - Feature extraction from pre-trained networks")
    print("   - Domain adaptation techniques")
    
    print("\n2. Data Augmentation:")
    print("   - Rotation, scaling, translation")
    print("   - Color jittering, noise injection")
    print("   - Mixup and CutMix techniques")
    
    print("\n3. Regularization:")
    print("   - Dropout for convolutional layers")
    print("   - Batch normalization")
    print("   - Weight decay and early stopping")
    
    print("\n4. Attention Mechanisms:")
    print("   - Self-attention in vision")
    print("   - Squeeze-and-excitation blocks")
    print("   - Channel and spatial attention")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of CNNs")
    print("="*50)
    print("1. Image Classification:")
    print("   - Object recognition in images")
    print("   - Medical image analysis")
    print("   - Satellite image classification")
    
    print("\n2. Object Detection:")
    print("   - YOLO, R-CNN, SSD architectures")
    print("   - Real-time object detection")
    print("   - Instance segmentation")
    
    print("\n3. Semantic Segmentation:")
    print("   - Pixel-wise classification")
    print("   - U-Net, SegNet, DeepLab")
    print("   - Medical image segmentation")
    
    print("\n4. Generative Models:")
    print("   - GANs for image synthesis")
    print("   - Style transfer")
    print("   - Super-resolution")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for CNNs")
    print("="*50)
    print("1. Architecture Design:")
    print("   - Start with proven architectures")
    print("   - Consider computational constraints")
    print("   - Balance depth and width")
    
    print("\n2. Training Strategy:")
    print("   - Use appropriate learning rates")
    print("   - Implement learning rate scheduling")
    print("   - Monitor for overfitting")
    
    print("\n3. Data Handling:")
    print("   - Apply proper data augmentation")
    print("   - Ensure data quality and diversity")
    print("   - Handle class imbalance")
    
    print("\n4. Evaluation:")
    print("   - Use multiple evaluation metrics")
    print("   - Validate on independent test sets")
    print("   - Consider domain-specific requirements")
    
    print("\n5. Optimization:")
    print("   - Use GPU acceleration when possible")
    print("   - Implement efficient data loading")
    print("   - Consider model compression techniques")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow/Keras: High-level CNN APIs")
    print("- PyTorch: Flexible deep learning framework")
    print("- OpenCV: Computer vision preprocessing")
    print("- These provide GPU acceleration and optimized implementations")