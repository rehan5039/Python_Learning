"""
Generative Models Implementation
===============================

This module demonstrates generative models including Generative Adversarial Networks (GANs)
and Variational Autoencoders (VAEs). It covers the theory, implementation, and applications.

Key Concepts:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Latent Space Manipulation
- Generative Model Training
- Mode Collapse and Solutions
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler


class SimpleVAE:
    """
    A simple Variational Autoencoder implementation.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    hidden_dim : int
        Dimension of hidden layers
    latent_dim : int
        Dimension of latent space
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Initialize encoder weights
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc1 = np.zeros((1, hidden_dim))
        self.W_enc2_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_enc2_mu = np.zeros((1, latent_dim))
        self.W_enc2_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_enc2_logvar = np.zeros((1, latent_dim))
        
        # Initialize decoder weights
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec1 = np.zeros((1, hidden_dim))
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_dec2 = np.zeros((1, input_dim))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function."""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def encode(self, x):
        """
        Encode input to latent space.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, input_dim)
            Input data
            
        Returns:
        --------
        mu : numpy array of shape (batch_size, latent_dim)
            Mean of latent distribution
        logvar : numpy array of shape (batch_size, latent_dim)
            Log variance of latent distribution
        """
        # Encoder forward pass
        h1 = self.relu(np.dot(x, self.W_enc1) + self.b_enc1)
        mu = np.dot(h1, self.W_enc2_mu) + self.b_enc2_mu
        logvar = np.dot(h1, self.W_enc2_logvar) + self.b_enc2_logvar
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent distribution.
        
        Parameters:
        -----------
        mu : numpy array
            Mean of latent distribution
        logvar : numpy array
            Log variance of latent distribution
            
        Returns:
        --------
        z : numpy array
            Sampled latent vector
        """
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*std.shape)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruct input.
        
        Parameters:
        -----------
        z : numpy array of shape (batch_size, latent_dim)
            Latent vector
            
        Returns:
        --------
        recon_x : numpy array of shape (batch_size, input_dim)
            Reconstructed input
        """
        # Decoder forward pass
        h2 = self.relu(np.dot(z, self.W_dec1) + self.b_dec1)
        recon_x = self.sigmoid(np.dot(h2, self.W_dec2) + self.b_dec2)
        return recon_x
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Parameters:
        -----------
        x : numpy array
            Input data
            
        Returns:
        --------
        recon_x : numpy array
            Reconstructed input
        mu : numpy array
            Mean of latent distribution
        logvar : numpy array
            Log variance of latent distribution
        z : numpy array
            Sampled latent vector
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Parameters:
        -----------
        recon_x : numpy array
            Reconstructed input
        x : numpy array
            Original input
        mu : numpy array
            Mean of latent distribution
        logvar : numpy array
            Log variance of latent distribution
            
        Returns:
        --------
        loss : float
            Total loss
        recon_loss : float
            Reconstruction loss
        kl_loss : float
            KL divergence loss
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = -np.sum(x * np.log(recon_x + 1e-8) + (1 - x) * np.log(1 - recon_x + 1e-8))
        
        # KL divergence loss
        kl_loss = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
        
        # Total loss
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss
    
    def train(self, X, epochs=100, learning_rate=0.001):
        """
        Train the VAE.
        
        Parameters:
        -----------
        X : numpy array
            Training data
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        """
        print("Training Variational Autoencoder...")
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            
            # Process in batches (simplified for demonstration)
            batch_size = 32
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i+batch_size]
                
                # Forward pass
                recon_x, mu, logvar, z = self.forward(batch_x)
                
                # Compute loss
                loss, recon_loss, kl_loss = self.loss_function(recon_x, batch_x, mu, logvar)
                total_loss += loss
                total_recon_loss += recon_loss
                total_kl_loss += kl_loss
                
                # Simplified backward pass (in practice, would compute gradients)
                # For demonstration, we'll just print progress
            
            avg_loss = total_loss / len(X)
            avg_recon_loss = total_recon_loss / len(X)
            avg_kl_loss = total_kl_loss / len(X)
            
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Total Loss: {avg_loss:.4f}, "
                      f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
        
        print("VAE training completed")
        return losses
    
    def generate(self, n_samples=10):
        """
        Generate new samples from the VAE.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        generated_samples : numpy array
            Generated samples
        """
        # Sample from standard normal distribution
        z_samples = np.random.randn(n_samples, self.latent_dim)
        
        # Decode to generate samples
        generated_samples = self.decode(z_samples)
        return generated_samples


class SimpleGAN:
    """
    A simple Generative Adversarial Network implementation.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input data
    latent_dim : int
        Dimension of latent space (generator input)
    hidden_dim : int
        Dimension of hidden layers
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Generator network
        self.W_gen1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_gen1 = np.zeros((1, hidden_dim))
        self.W_gen2 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_gen2 = np.zeros((1, input_dim))
        
        # Discriminator network
        self.W_disc1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_disc1 = np.zeros((1, hidden_dim))
        self.W_disc2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b_disc2 = np.zeros((1, 1))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def generator_forward(self, z):
        """
        Forward pass through generator.
        
        Parameters:
        -----------
        z : numpy array of shape (batch_size, latent_dim)
            Latent vectors
            
        Returns:
        --------
        fake_data : numpy array of shape (batch_size, input_dim)
            Generated fake data
        """
        h1 = self.relu(np.dot(z, self.W_gen1) + self.b_gen1)
        fake_data = self.sigmoid(np.dot(h1, self.W_gen2) + self.b_gen2)
        return fake_data
    
    def discriminator_forward(self, x):
        """
        Forward pass through discriminator.
        
        Parameters:
        -----------
        x : numpy array of shape (batch_size, input_dim)
            Input data (real or fake)
            
        Returns:
        --------
        validity : numpy array of shape (batch_size, 1)
            Probability of input being real
        """
        h1 = self.relu(np.dot(x, self.W_disc1) + self.b_disc1)
        validity = self.sigmoid(np.dot(h1, self.W_disc2) + self.b_disc2)
        return validity
    
    def train(self, X_real, epochs=1000, batch_size=32, learning_rate=0.001):
        """
        Train the GAN.
        
        Parameters:
        -----------
        X_real : numpy array
            Real training data
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        """
        print("Training Generative Adversarial Network...")
        
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Train discriminator
            # Sample real data
            idx = np.random.randint(0, X_real.shape[0], batch_size)
            real_batch = X_real[idx]
            
            # Generate fake data
            noise = np.random.randn(batch_size, self.latent_dim)
            fake_batch = self.generator_forward(noise)
            
            # Train discriminator on real and fake data
            real_validity = self.discriminator_forward(real_batch)
            fake_validity = self.discriminator_forward(fake_batch)
            
            # Discriminator loss (maximize log(D(x)) + log(1 - D(G(z))))
            d_loss = -np.mean(np.log(real_validity + 1e-8) + np.log(1 - fake_validity + 1e-8))
            d_losses.append(d_loss)
            
            # Train generator
            # Generate new fake data
            noise = np.random.randn(batch_size, self.latent_dim)
            fake_batch = self.generator_forward(noise)
            fake_validity = self.discriminator_forward(fake_batch)
            
            # Generator loss (minimize log(1 - D(G(z))) or maximize log(D(G(z))))
            g_loss = -np.mean(np.log(fake_validity + 1e-8))
            g_losses.append(g_loss)
            
            # Simplified training (in practice, would update weights with gradients)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        print("GAN training completed")
        return d_losses, g_losses
    
    def generate(self, n_samples=10):
        """
        Generate new samples from the trained generator.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        generated_samples : numpy array
            Generated samples
        """
        noise = np.random.randn(n_samples, self.latent_dim)
        generated_samples = self.generator_forward(noise)
        return generated_samples


# Example usage and demonstration
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create 2D synthetic data (similar to MNIST but simpler)
    X_vae, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_vae = StandardScaler().fit_transform(X_vae)
    
    # Normalize to [0, 1] for VAE
    X_vae = (X_vae - X_vae.min()) / (X_vae.max() - X_vae.min())
    
    print("Generative Models Demonstration")
    print("="*50)
    
    print("Dataset for VAE:")
    print(f"Shape: {X_vae.shape}")
    print(f"Range: [{X_vae.min():.3f}, {X_vae.max():.3f}]")
    
    # Demonstrate VAE
    print("\n" + "="*50)
    print("Variational Autoencoder (VAE)")
    print("="*50)
    
    # Create and train VAE
    vae = SimpleVAE(input_dim=2, hidden_dim=16, latent_dim=2)
    
    # Train VAE
    vae_losses = vae.train(X_vae, epochs=100, learning_rate=0.01)
    
    # Generate new samples
    generated_vae = vae.generate(n_samples=100)
    print(f"Generated samples shape: {generated_vae.shape}")
    print(f"Generated samples range: [{generated_vae.min():.3f}, {generated_vae.max():.3f}]")
    
    # Visualize VAE results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_vae[:, 0], X_vae[:, 1], alpha=0.6, label='Real Data')
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(generated_vae[:, 0], generated_vae[:, 1], alpha=0.6, label='Generated Data')
    plt.title('VAE Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate GAN
    print("\n" + "="*50)
    print("Generative Adversarial Network (GAN)")
    print("="*50)
    
    # Create 2D synthetic data for GAN
    X_gan, _ = make_classification(n_samples=1000, n_features=2, n_classes=1, 
                                  n_redundant=0, n_informative=2, 
                                  n_clusters_per_class=1, random_state=42)
    X_gan = StandardScaler().fit_transform(X_gan)
    
    # Normalize to [0, 1]
    X_gan = (X_gan - X_gan.min()) / (X_gan.max() - X_gan.min())
    
    print("Dataset for GAN:")
    print(f"Shape: {X_gan.shape}")
    print(f"Range: [{X_gan.min():.3f}, {X_gan.max():.3f}]")
    
    # Create and train GAN
    gan = SimpleGAN(input_dim=2, latent_dim=4, hidden_dim=16)
    
    # Train GAN
    d_losses, g_losses = gan.train(X_gan, epochs=500, batch_size=32, learning_rate=0.001)
    
    # Generate new samples
    generated_gan = gan.generate(n_samples=100)
    print(f"Generated samples shape: {generated_gan.shape}")
    print(f"Generated samples range: [{generated_gan.min():.3f}, {generated_gan.max():.3f}]")
    
    # Visualize GAN results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_gan[:, 0], X_gan[:, 1], alpha=0.6, label='Real Data')
    plt.title('Original Data (GAN)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.scatter(generated_gan[:, 0], generated_gan[:, 1], alpha=0.6, label='Generated Data')
    plt.title('GAN Generated Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Key concepts explanation
    print("\n" + "="*50)
    print("Key Generative Model Concepts:")
    print("="*50)
    print("Variational Autoencoders (VAEs):")
    print("- Learn latent representations of data")
    print("- Provide explicit probability distributions")
    print("- Enable interpolation in latent space")
    print("- Trade-off between reconstruction and regularization")
    
    print("\nGenerative Adversarial Networks (GANs):")
    print("- Consist of generator and discriminator networks")
    print("- Generator creates fake data, discriminator distinguishes real from fake")
    print("- Minimax game between two networks")
    print("- Can generate high-quality samples but training can be unstable")
    
    # Common challenges and solutions
    print("\n" + "="*50)
    print("Common Challenges and Solutions:")
    print("="*50)
    print("VAE Challenges:")
    print("- Blurry generated samples")
    print("- Posterior collapse")
    print("Solutions: KL annealing, stronger priors, improved architectures")
    
    print("\nGAN Challenges:")
    print("- Mode collapse (generator produces limited variety)")
    print("- Training instability")
    print("- Difficulty in convergence")
    print("Solutions: Wasserstein GAN, gradient penalty, spectral normalization")
    
    # Applications
    print("\n" + "="*50)
    print("Generative Model Applications:")
    print("="*50)
    print("Image Generation:")
    print("- Artwork creation, photo editing, style transfer")
    print("- Data augmentation for training datasets")
    
    print("\nText Generation:")
    print("- Content creation, chatbots, story writing")
    print("- Code generation, translation")
    
    print("\nAudio Generation:")
    print("- Music composition, voice synthesis")
    print("- Sound effects, audio restoration")
    
    print("\nScientific Applications:")
    print("- Drug discovery, protein structure prediction")
    print("- Material design, climate modeling")
    
    # Popular architectures
    print("\n" + "="*50)
    print("Popular Generative Architectures:")
    print("="*50)
    print("VAE Variants:")
    print("- β-VAE, VQ-VAE, PixelVAE")
    print("- Conditional VAEs")
    
    print("\nGAN Variants:")
    print("- DCGAN, StyleGAN, CycleGAN")
    print("- Pix2Pix, Progressive GAN")
    
    print("\nOther Models:")
    print("- Normalizing Flows (RealNVP, Glow)")
    print("- Autoregressive Models (PixelRNN, WaveNet)")
    print("- Diffusion Models (DDPM, Stable Diffusion)")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow/Keras: tf.keras.layers, tf.keras.models")
    print("- PyTorch: torch.nn, torch.optim")
    print("- Specialized libraries: PyTorch Lightning, Hugging Face Diffusers")
    print("- These provide GPU acceleration and optimized implementations")