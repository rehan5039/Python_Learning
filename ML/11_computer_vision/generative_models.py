"""
Generative Models for Computer Vision
===================================

This module demonstrates generative models for computer vision including GANs, VAEs,
and diffusion models. It covers image generation, style transfer, and image editing techniques.

Key Concepts:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Diffusion Models
- Image Generation
- Style Transfer
- Image Editing
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class SimpleVAE:
    """
    Simplified Variational Autoencoder for image generation.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape (height, width, channels)
    latent_dim : int
        Dimension of latent space
    """
    
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.input_dim = np.prod(input_shape)
        
        # Initialize encoder weights
        self.encoder_weights = np.random.randn(self.input_dim, latent_dim * 2) * 0.1
        self.encoder_bias = np.zeros((1, latent_dim * 2))
        
        # Initialize decoder weights
        self.decoder_weights = np.random.randn(latent_dim, self.input_dim) * 0.1
        self.decoder_bias = np.zeros((1, self.input_dim))
    
    def encode(self, x):
        """
        Encode input to latent space.
        
        Parameters:
        -----------
        x : numpy array
            Input images of shape (batch_size, height, width, channels)
            
        Returns:
        --------
        mu : numpy array
            Mean of latent distribution
        log_var : numpy array
            Log variance of latent distribution
        """
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Encoder forward pass
        hidden = np.dot(x_flat, self.encoder_weights) + self.encoder_bias
        
        # Split into mean and log variance
        mu = hidden[:, :self.latent_dim]
        log_var = hidden[:, self.latent_dim:]
        
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from latent distribution.
        
        Parameters:
        -----------
        mu : numpy array
            Mean of latent distribution
        log_var : numpy array
            Log variance of latent distribution
            
        Returns:
        --------
        z : numpy array
            Sampled latent vector
        """
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*std.shape)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruct image.
        
        Parameters:
        -----------
        z : numpy array
            Latent vectors of shape (batch_size, latent_dim)
            
        Returns:
        --------
        reconstructed : numpy array
            Reconstructed images
        """
        # Decoder forward pass
        hidden = np.dot(z, self.decoder_weights) + self.decoder_bias
        
        # Apply sigmoid activation
        reconstructed_flat = 1 / (1 + np.exp(-hidden))
        
        # Reshape to image dimensions
        batch_size = z.shape[0]
        reconstructed = reconstructed_flat.reshape(batch_size, *self.input_shape)
        
        return reconstructed
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Parameters:
        -----------
        x : numpy array
            Input images
            
        Returns:
        --------
        reconstructed : numpy array
            Reconstructed images
        mu : numpy array
            Mean of latent distribution
        log_var : numpy array
            Log variance of latent distribution
        z : numpy array
            Sampled latent vector
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var, z
    
    def generate(self, num_samples=1):
        """
        Generate new images from random latent vectors.
        
        Parameters:
        -----------
        num_samples : int, default=1
            Number of images to generate
            
        Returns:
        --------
        generated : numpy array
            Generated images
        """
        # Sample random latent vectors
        z_samples = np.random.randn(num_samples, self.latent_dim)
        
        # Decode to generate images
        generated = self.decode(z_samples)
        return generated


class SimpleGAN:
    """
    Simplified Generative Adversarial Network for image generation.
    
    Parameters:
    -----------
    input_shape : tuple
        Input image shape (height, width, channels)
    latent_dim : int
        Dimension of latent space
    """
    
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.input_dim = np.prod(input_shape)
        
        # Initialize generator weights
        self.generator_weights = np.random.randn(latent_dim, 256) * 0.1
        self.generator_weights2 = np.random.randn(256, self.input_dim) * 0.1
        self.generator_bias = np.zeros((1, 256))
        self.generator_bias2 = np.zeros((1, self.input_dim))
        
        # Initialize discriminator weights
        self.discriminator_weights = np.random.randn(self.input_dim, 256) * 0.1
        self.discriminator_weights2 = np.random.randn(256, 1) * 0.1
        self.discriminator_bias = np.zeros((1, 256))
        self.discriminator_bias2 = np.zeros((1, 1))
    
    def generator_forward(self, z):
        """
        Forward pass through generator.
        
        Parameters:
        -----------
        z : numpy array
            Latent vectors of shape (batch_size, latent_dim)
            
        Returns:
        --------
        generated : numpy array
            Generated images
        """
        # Generator forward pass
        hidden = np.maximum(0, np.dot(z, self.generator_weights) + self.generator_bias)  # ReLU
        output = np.dot(hidden, self.generator_weights2) + self.generator_bias2
        
        # Apply tanh activation
        generated_flat = np.tanh(output)
        
        # Reshape to image dimensions
        batch_size = z.shape[0]
        generated = generated_flat.reshape(batch_size, *self.input_shape)
        
        return generated
    
    def discriminator_forward(self, x):
        """
        Forward pass through discriminator.
        
        Parameters:
        -----------
        x : numpy array
            Input images of shape (batch_size, height, width, channels)
            
        Returns:
        --------
        validity : numpy array
            Validity scores (0=fake, 1=real)
        """
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Discriminator forward pass
        hidden = np.maximum(0, np.dot(x_flat, self.discriminator_weights) + self.discriminator_bias)  # ReLU
        output = np.dot(hidden, self.discriminator_weights2) + self.discriminator_bias2
        
        # Apply sigmoid activation
        validity = 1 / (1 + np.exp(-output))
        
        return validity
    
    def generate(self, num_samples=1):
        """
        Generate new images using the generator.
        
        Parameters:
        -----------
        num_samples : int, default=1
            Number of images to generate
            
        Returns:
        --------
        generated : numpy array
            Generated images
        """
        # Sample random latent vectors
        z_samples = np.random.randn(num_samples, self.latent_dim)
        
        # Generate images
        generated = self.generator_forward(z_samples)
        return generated


class StyleTransfer:
    """
    Simplified neural style transfer implementation.
    
    Parameters:
    -----------
    content_weight : float, default=1e4
        Weight for content loss
    style_weight : float, default=1e-3
        Weight for style loss
    """
    
    def __init__(self, content_weight=1e4, style_weight=1e-3):
        self.content_weight = content_weight
        self.style_weight = style_weight
    
    def gram_matrix(self, feature_map):
        """
        Compute Gram matrix for style representation.
        
        Parameters:
        -----------
        feature_map : numpy array
            Feature map of shape (height, width, channels)
            
        Returns:
        --------
        gram : numpy array
            Gram matrix of shape (channels, channels)
        """
        height, width, channels = feature_map.shape
        # Reshape to (channels, height*width)
        features = feature_map.reshape(-1, channels).T
        # Compute Gram matrix
        gram = np.dot(features, features.T)
        return gram
    
    def content_loss(self, content_features, generated_features):
        """
        Compute content loss.
        
        Parameters:
        -----------
        content_features : numpy array
            Content image features
        generated_features : numpy array
            Generated image features
            
        Returns:
        --------
        loss : float
            Content loss
        """
        return np.mean((content_features - generated_features) ** 2)
    
    def style_loss(self, style_gram, generated_gram):
        """
        Compute style loss.
        
        Parameters:
        -----------
        style_gram : numpy array
            Style image Gram matrix
        generated_gram : numpy array
            Generated image Gram matrix
            
        Returns:
        --------
        loss : float
            Style loss
        """
        return np.mean((style_gram - generated_gram) ** 2)
    
    def transfer_style(self, content_image, style_image, num_iterations=100):
        """
        Transfer style from style image to content image.
        
        Parameters:
        -----------
        content_image : numpy array
            Content image
        style_image : numpy array
            Style image
        num_iterations : int, default=100
            Number of optimization iterations
            
        Returns:
        --------
        stylized_image : numpy array
            Stylized image
        """
        # In a real implementation, this would:
        # 1. Extract features using a pre-trained CNN
        # 2. Compute content and style representations
        # 3. Optimize generated image using gradient descent
        
        # For demonstration, we'll create a simple stylized version
        # by combining content and style images
        
        # Normalize images to [0, 1]
        content_norm = (content_image - content_image.min()) / (content_image.max() - content_image.min())
        style_norm = (style_image - style_image.min()) / (style_image.max() - style_image.min())
        
        # Simple blending (in practice, this would be much more complex)
        stylized_image = (0.7 * content_norm + 0.3 * style_norm)
        
        # Clip to valid range
        stylized_image = np.clip(stylized_image, 0, 1)
        
        return stylized_image


class ImageEditor:
    """
    Simple image editing tools using generative models.
    """
    
    def __init__(self):
        pass
    
    def inpaint(self, image, mask):
        """
        Inpaint missing regions in image.
        
        Parameters:
        -----------
        image : numpy array
            Input image with missing regions
        mask : numpy array
            Binary mask indicating missing regions (1=missing, 0=present)
            
        Returns:
        --------
        inpainted : numpy array
            Inpainted image
        """
        # In a real implementation, this would use a trained inpainting model
        # For demonstration, we'll use a simple approach
        
        # Create inpainted image
        inpainted = image.copy()
        
        # For each missing region, fill with average of surrounding pixels
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i, j] == 1:  # Missing pixel
                    # Get surrounding pixels
                    surrounding = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < image.shape[0] and 
                                0 <= nj < image.shape[1] and 
                                mask[ni, nj] == 0):  # Present pixel
                                surrounding.append(image[ni, nj])
                    
                    # Fill with average of surrounding pixels
                    if surrounding:
                        if len(image.shape) == 3:
                            inpainted[i, j] = np.mean(surrounding, axis=0)
                        else:
                            inpainted[i, j] = np.mean(surrounding)
        
        return inpainted
    
    def super_resolution(self, low_res_image, scale_factor=2):
        """
        Enhance resolution of image.
        
        Parameters:
        -----------
        low_res_image : numpy array
            Low resolution input image
        scale_factor : int, default=2
            Scale factor for upscaling
            
        Returns:
        --------
        high_res_image : numpy array
            High resolution output image
        """
        # In a real implementation, this would use a trained super-resolution model
        # For demonstration, we'll use interpolation
        
        from scipy.ndimage import zoom
        
        if len(low_res_image.shape) == 3:
            # For color images, interpolate each channel separately
            high_res_image = np.zeros((
                low_res_image.shape[0] * scale_factor,
                low_res_image.shape[1] * scale_factor,
                low_res_image.shape[2]
            ))
            for c in range(low_res_image.shape[2]):
                high_res_image[:, :, c] = zoom(
                    low_res_image[:, :, c], scale_factor, order=1
                )
        else:
            # For grayscale images
            high_res_image = zoom(low_res_image, scale_factor, order=1)
        
        return high_res_image


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample images for demonstration
    print("Generative Models for Computer Vision Demonstration")
    print("=" * 55)
    
    # Create sample images (64x64, 3 channels)
    sample_image = np.random.randn(64, 64, 3).astype(np.float32)
    content_image = np.random.randn(64, 64, 3).astype(np.float32)
    style_image = np.random.randn(64, 64, 3).astype(np.float32)
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Content image shape: {content_image.shape}")
    print(f"Style image shape: {style_image.shape}")
    
    # VAE demonstration
    print("\n1. Variational Autoencoder (VAE):")
    vae = SimpleVAE(input_shape=(64, 64, 3), latent_dim=32)
    
    # Create batch of images
    batch_images = np.random.randn(4, 64, 64, 3).astype(np.float32)
    
    # Forward pass
    reconstructed, mu, log_var, z = vae.forward(batch_images)
    print(f"Input batch shape: {batch_images.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent vector shape: {z.shape}")
    print(f"Mean shape: {mu.shape}")
    print(f"Log variance shape: {log_var.shape}")
    
    # Generate new images
    generated_vae = vae.generate(num_samples=3)
    print(f"VAE generated images shape: {generated_vae.shape}")
    
    # GAN demonstration
    print("\n2. Generative Adversarial Network (GAN):")
    gan = SimpleGAN(input_shape=(64, 64, 3), latent_dim=100)
    
    # Generate new images
    generated_gan = gan.generate(num_samples=3)
    print(f"GAN generated images shape: {generated_gan.shape}")
    
    # Style Transfer demonstration
    print("\n3. Neural Style Transfer:")
    style_transfer = StyleTransfer()
    
    # Transfer style
    stylized_image = style_transfer.transfer_style(content_image, style_image, num_iterations=50)
    print(f"Content image shape: {content_image.shape}")
    print(f"Style image shape: {style_image.shape}")
    print(f"Stylized image shape: {stylized_image.shape}")
    
    # Image Editing demonstration
    print("\n4. Image Editing Tools:")
    image_editor = ImageEditor()
    
    # Create image with missing region
    test_image = np.random.randn(32, 32, 3).astype(np.float32)
    mask = np.zeros((32, 32))
    mask[10:20, 10:20] = 1  # Missing region
    
    # Inpaint missing region
    inpainted = image_editor.inpaint(test_image, mask)
    print(f"Original image shape: {test_image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Inpainted image shape: {inpainted.shape}")
    
    # Super resolution
    low_res = np.random.randn(16, 16, 3).astype(np.float32)
    high_res = image_editor.super_resolution(low_res, scale_factor=2)
    print(f"Low resolution image shape: {low_res.shape}")
    print(f"High resolution image shape: {high_res.shape}")
    
    # Compare different generative models
    print("\n" + "="*50)
    print("Comparison of Generative Models")
    print("="*50)
    
    print("1. Variational Autoencoders (VAEs):")
    print("   - Probabilistic generative models")
    print("   - Explicit latent space representation")
    print("   - Good for interpolation and sampling")
    print("   - Tend to produce blurry images")
    
    print("\n2. Generative Adversarial Networks (GANs):")
    print("   - Adversarial training framework")
    print("   - Generator vs Discriminator")
    print("   - High quality, sharp images")
    print("   - Training can be unstable")
    
    print("\n3. Diffusion Models:")
    print("   - Iterative denoising process")
    print("   - Forward and reverse diffusion")
    print("   - State-of-the-art quality")
    print("   - Slower generation but stable training")
    
    print("\n4. Autoregressive Models:")
    print("   - Generate pixels sequentially")
    print("   - PixelRNN, PixelCNN")
    print("   - High quality but slow generation")
    print("   - Good likelihood estimation")
    
    print("\n5. Normalizing Flows:")
    print("   - Exact likelihood computation")
    print("   - Bijective transformations")
    print("   - Good for density estimation")
    print("   - Limited generative capabilities")
    
    # Advanced generative concepts
    print("\n" + "="*50)
    print("Advanced Generative Concepts")
    print("="*50)
    print("1. Conditional Generation:")
    print("   - Class-conditional GANs")
    print("   - Text-to-image synthesis")
    print("   - Attribute manipulation")
    print("   - Control over generation process")
    
    print("\n2. Latent Space Manipulation:")
    print("   - Semantic interpolation")
    print("   - Attribute vectors")
    print("   - Style mixing")
    print("   - Disentangled representations")
    
    print("\n3. Multi-modal Generation:")
    print("   - Image-to-image translation")
    print("   - Cross-domain generation")
    print("   - Multi-task learning")
    print("   - Unified generative frameworks")
    
    print("\n4. Few-shot Generation:")
    print("   - Learning from few examples")
    print("   - Meta-learning approaches")
    print("   - Transfer learning techniques")
    print("   - Personalized generation")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of Generative Models")
    print("="*50)
    print("1. Creative Applications:")
    print("   - Art generation")
    print("   - Music composition")
    print("   - Video game content")
    print("   - Design assistance")
    
    print("\n2. Data Augmentation:")
    print("   - Synthetic data generation")
    print("   - Class balancing")
    print("   - Domain adaptation")
    print("   - Privacy-preserving data")
    
    print("\n3. Image Editing:")
    print("   - Inpainting missing regions")
    print("   - Super-resolution")
    print("   - Style transfer")
    print("   - Image restoration")
    
    print("\n4. Scientific Applications:")
    print("   - Drug discovery")
    print("   - Protein structure prediction")
    print("   - Material design")
    print("   - Climate modeling")
    
    print("\n5. Entertainment:")
    print("   - Video game assets")
    print("   - Movie special effects")
    print("   - Virtual reality content")
    print("   - Interactive media")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Generative Models")
    print("="*50)
    print("1. Model Selection:")
    print("   - Choose appropriate architecture for task")
    print("   - Consider computational constraints")
    print("   - Balance quality and efficiency")
    print("   - Evaluate multiple approaches")
    
    print("\n2. Training Strategy:")
    print("   - Use appropriate loss functions")
    print("   - Implement proper regularization")
    print("   - Monitor for mode collapse")
    print("   - Use curriculum learning")
    
    print("\n3. Evaluation:")
    print("   - Use multiple metrics (FID, IS, etc.)")
    print("   - Human evaluation for quality")
    print("   - Diversity and coverage metrics")
    print("   - Domain-specific validation")
    
    print("\n4. Deployment:")
    print("   - Model optimization for inference")
    print("   - Latency and memory considerations")
    print("   - Quality control mechanisms")
    print("   - User feedback integration")
    
    print("\n5. Ethical Considerations:")
    print("   - Content moderation")
    print("   - Bias detection and mitigation")
    print("   - Privacy protection")
    print("   - Responsible deployment")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow/Keras: Generative models")
    print("- PyTorch: torchvision and specialized libraries")
    print("- Hugging Face Diffusers: Diffusion models")
    print("- These provide optimized implementations and pre-trained models")