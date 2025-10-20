"""
Practice Problems Solutions: Neural Networks

This module contains example solutions for the practice problems in Chapter 8.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..neural_networks.activation_functions import ActivationFunction, ReLU, LeakyReLU
from ..neural_networks.network_architectures import NeuralNetwork
from ..neural_networks.training_algorithms import GradientDescentOptimizer, AdamOptimizer, NetworkTrainer

# Problem 1: Activation Function Implementation
class Softplus(ActivationFunction):
    """Softplus activation function: log(1 + e^x)"""
    
    def __init__(self):
        super().__init__("Softplus")
    
    def forward(self, x):
        """Softplus function"""
        # For numerical stability
        return np.log(1 + np.exp(np.clip(x, -500, 500)))
    
    def derivative(self, x):
        """Derivative of softplus: sigmoid(x)"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

def problem_1_solution():
    """Solution for Problem 1: Activation Function Implementation"""
    print("Problem 1: Activation Function Implementation")
    print("=" * 45)
    
    # Create activation function instances
    relu = ReLU()
    leaky_relu = LeakyReLU(alpha=0.01)
    softplus = Softplus()
    
    # Test values
    x = np.linspace(-5, 5, 100)
    
    # Compute activations
    y_relu = relu.forward(x)
    y_leaky = leaky_relu.forward(x)
    y_softplus = softplus.forward(x)
    
    # Compute derivatives
    dy_relu = relu.derivative(x)
    dy_leaky = leaky_relu.derivative(x)
    dy_softplus = softplus.derivative(x)
    
    # Plot activation functions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_relu, label='ReLU', linewidth=2)
    plt.plot(x, y_leaky, label='Leaky ReLU', linewidth=2)
    plt.plot(x, y_softplus, label='Softplus', linewidth=2)
    plt.title('Activation Functions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x, dy_relu, label='ReLU Derivative', linewidth=2)
    plt.plot(x, dy_leaky, label='Leaky ReLU Derivative', linewidth=2)
    plt.plot(x, dy_softplus, label='Softplus Derivative', linewidth=2)
    plt.title('Derivatives')
    plt.xlabel('Input')
    plt.ylabel('Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Analysis:")
    print("1. ReLU: Simple and efficient, but suffers from dead neurons")
    print("2. Leaky ReLU: Addresses dead neuron problem with small negative slope")
    print("3. Softplus: Smooth approximation of ReLU, always differentiable")

# Problem 2: Network Architecture Design
def problem_2_solution():
    """Solution for Problem 2: Network Architecture Design"""
    print("\n\nProblem 2: Network Architecture Design")
    print("=" * 38)
    
    # Design for 28x28 grayscale images with 10 classes
    input_size = 28 * 28  # 784
    hidden_layers = [128, 64, 32]
    output_size = 10
    
    print("Network Architecture for Image Classification:")
    print(f"Input Layer: {input_size} neurons (28x28 pixels)")
    for i, size in enumerate(hidden_layers):
        print(f"Hidden Layer {i+1}: {size} neurons (ReLU activation)")
    print(f"Output Layer: {output_size} neurons (Softmax activation)")
    
    # Calculate total parameters
    total_params = 0
    layer_sizes = [input_size] + hidden_layers + [output_size]
    
    print("\nParameter Calculation:")
    for i in range(len(layer_sizes) - 1):
        weights = layer_sizes[i] * layer_sizes[i+1]
        biases = layer_sizes[i+1]
        layer_params = weights + biases
        total_params += layer_params
        print(f"Layer {i+1}: {weights} weights + {biases} biases = {layer_params} parameters")
    
    print(f"\nTotal Parameters: {total_params:,}")
    
    print("\nActivation Function Choices:")
    print("1. ReLU for hidden layers: Computationally efficient, helps with vanishing gradient")
    print("2. Softmax for output layer: Produces probability distribution over classes")
    
    print("\nRegularization Techniques:")
    print("1. Dropout: Randomly set neuron outputs to zero during training")
    print("2. L2 Regularization: Add penalty term to loss function")
    print("3. Batch Normalization: Normalize layer inputs to stabilize training")
    print("4. Early Stopping: Stop training when validation performance plateaus")

# Problem 3: Training Algorithm Comparison
class MomentumOptimizer:
    """Momentum-based Gradient Descent Optimizer"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None
    
    def update_parameters(self, weights, biases, dW, db):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]
        
        updated_weights = []
        updated_biases = []
        
        for i in range(len(weights)):
            # Update velocity
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + (1 - self.momentum) * dW[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + (1 - self.momentum) * db[i]
            
            # Update parameters
            w = weights[i] - self.learning_rate * self.velocity_w[i]
            b = biases[i] - self.learning_rate * self.velocity_b[i]
            
            updated_weights.append(w)
            updated_biases.append(b)
        
        return updated_weights, updated_biases

def problem_3_solution():
    """Solution for Problem 3: Training Algorithm Comparison"""
    print("\n\nProblem 3: Training Algorithm Comparison")
    print("=" * 40)
    
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create networks for each optimizer
    networks = {
        'Gradient Descent': NeuralNetwork([2, 4, 1], ['relu', 'sigmoid']),
        'Momentum': NeuralNetwork([2, 4, 1], ['relu', 'sigmoid']),
        'Adam': NeuralNetwork([2, 4, 1], ['relu', 'sigmoid'])
    }
    
    # Create trainers with different optimizers
    trainers = {
        'Gradient Descent': NetworkTrainer(networks['Gradient Descent'], GradientDescentOptimizer(0.1)),
        'Momentum': NetworkTrainer(networks['Momentum'], MomentumOptimizer(0.1, 0.9)),
        'Adam': NetworkTrainer(networks['Adam'], AdamOptimizer(0.1))
    }
    
    # Train networks
    loss_histories = {}
    for name, trainer in trainers.items():
        print(f"Training with {name}...")
        trainer.train(X, y, epochs=1000, verbose=False)
        loss_histories[name] = trainer.loss_history
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for name, loss_history in loss_histories.items():
        plt.plot(loss_history, label=name, linewidth=2)
    plt.title('Optimizer Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Analysis:")
    print("1. Gradient Descent: Simple but may get stuck in local minima")
    print("2. Momentum: Helps accelerate convergence and escape local minima")
    print("3. Adam: Combines momentum with adaptive learning rates, often most effective")

# Problem 4: Backpropagation Implementation
def gradient_checking(network, X, y, epsilon=1e-7):
    """Simple gradient checking implementation"""
    # Forward pass
    output, activations = network.forward(X)
    
    # Compute analytical gradients
    trainer = NetworkTrainer(network)
    dW_analytical, db_analytical = trainer.compute_gradients(X, y, activations)
    
    # Compute numerical gradients (simplified)
    numerical_gradients = []
    for i in range(len(network.weights)):
        grad = np.zeros_like(network.weights[i])
        for j in range(network.weights[i].shape[0]):
            for k in range(network.weights[i].shape[1]):
                # Add epsilon
                network.weights[i][j, k] += epsilon
                loss_plus, _ = network.forward(X)
                loss_plus = np.mean((loss_plus - y) ** 2)
                
                # Subtract epsilon
                network.weights[i][j, k] -= 2 * epsilon
                loss_minus, _ = network.forward(X)
                loss_minus = np.mean((loss_minus - y) ** 2)
                
                # Restore original value
                network.weights[i][j, k] += epsilon
                
                # Compute numerical gradient
                grad[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
        numerical_gradients.append(grad)
    
    # Compare gradients
    differences = []
    for i in range(len(dW_analytical)):
        diff = np.abs(dW_analytical[i] - numerical_gradients[i])
        differences.append(np.mean(diff))
    
    return differences

def problem_4_solution():
    """Solution for Problem 4: Backpropagation Implementation"""
    print("\n\nProblem 4: Backpropagation Implementation")
    print("=" * 40)
    
    # Create a simple network
    network = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])
    
    # Sample data
    X = np.array([[0.5, 0.3]])
    y = np.array([[0.8]])
    
    print("Performing gradient checking...")
    
    # Perform gradient checking
    differences = gradient_checking(network, X, y)
    
    print("Gradient checking results:")
    for i, diff in enumerate(differences):
        print(f"Layer {i+1} gradient difference: {diff:.2e}")
    
    if all(diff < 1e-5 for diff in differences):
        print("✓ Gradients verified - implementation is likely correct")
    else:
        print("✗ Gradient mismatch - implementation may have errors")
    
    print("\nBackpropagation Implementation Notes:")
    print("1. Chain rule is applied from output to input layers")
    print("2. Each layer's error is computed based on the next layer's error")
    print("3. Activation function derivatives are crucial for correct gradients")
    print("4. Matrix dimensions must be carefully maintained throughout")

# Problem 5: Practical Network Training
def problem_5_solution():
    """Solution for Problem 5: Practical Network Training"""
    print("\n\nProblem 5: Practical Network Training")
    print("=" * 35)
    
    # Generate synthetic dataset
    np.random.seed(42)
    X = np.random.randn(1000, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    # Split into train/validation/test
    train_size = 700
    val_size = 150
    
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {len(X_test)} test samples")
    
    # Create network
    network = NeuralNetwork([4, 8, 4, 1], ['relu', 'relu', 'sigmoid'])
    
    # Train with early stopping
    trainer = NetworkTrainer(network, AdamOptimizer(0.01))
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Training with early stopping...")
    for epoch in range(100):
        # Train one epoch
        output, activations = network.forward(X_train)
        loss = trainer.compute_loss(y_train, output)
        dW, db = trainer.compute_gradients(X_train, y_train, activations)
        network.weights, network.biases = trainer.optimizer.update_parameters(
            network.weights, network.biases, dW, db
        )
        
        # Validate
        val_output, _ = network.forward(X_val)
        val_loss = trainer.compute_loss(y_val, val_output)
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate on test set
    test_output, _ = network.forward(X_test)
    test_predictions = (test_output > 0.5).astype(int)
    accuracy = np.mean(test_predictions == y_test)
    
    print(f"Test Accuracy: {accuracy:.3f}")
    
    print("\nHyperparameter Tuning Approach:")
    print("1. Learning Rate: Try [0.001, 0.01, 0.1] and use validation performance")
    print("2. Network Architecture: Experiment with different layer sizes")
    print("3. Regularization: Adjust dropout rates and L2 penalty coefficients")
    print("4. Batch Size: Try different sizes [16, 32, 64, 128]")

# Run all solutions
if __name__ == "__main__":
    problem_1_solution()
    problem_2_solution()
    problem_3_solution()
    problem_4_solution()
    problem_5_solution()