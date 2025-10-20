"""
Image Preprocessing Implementation
================================

This module demonstrates various image preprocessing techniques essential for computer vision tasks.
It covers image loading, resizing, normalization, and enhancement techniques.

Key Concepts:
- Image Loading and Conversion
- Resizing and Scaling
- Normalization and Standardization
- Noise Reduction
- Geometric Transformations
- Color Space Conversion
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage


class ImagePreprocessor:
    """
    A comprehensive image preprocessor for computer vision tasks.
    
    Parameters:
    -----------
    target_size : tuple, optional
        Target size for resizing (height, width)
    normalize : bool, default=True
        Whether to normalize pixel values
    color_mode : str, default='rgb'
        Color mode ('rgb', 'grayscale', 'hsv')
    """
    
    def __init__(self, target_size=None, normalize=True, color_mode='rgb'):
        self.target_size = target_size
        self.normalize = normalize
        self.color_mode = color_mode
    
    def load_image(self, image_path):
        """
        Load an image from file.
        
        Parameters:
        -----------
        image_path : str
            Path to image file
            
        Returns:
        --------
        image : numpy array
            Loaded image array
        """
        try:
            image = Image.open(image_path)
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def resize_image(self, image, size=None):
        """
        Resize an image.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        size : tuple, optional
            Target size (height, width)
            
        Returns:
        --------
        resized_image : numpy array
            Resized image
        """
        if size is None:
            size = self.target_size
        
        if size is None:
            return image
        
        # Convert to PIL Image for resizing
        pil_image = Image.fromarray(image)
        resized_pil = pil_image.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        return np.array(resized_pil)
    
    def convert_color_space(self, image, target_mode=None):
        """
        Convert image color space.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        target_mode : str, optional
            Target color mode
            
        Returns:
        --------
        converted_image : numpy array
            Color space converted image
        """
        if target_mode is None:
            target_mode = self.color_mode
        
        if target_mode == 'grayscale' and len(image.shape) == 3:
            # Convert RGB to grayscale
            if image.shape[2] == 3:
                # Standard RGB to grayscale conversion
                converted = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                return converted.astype(np.uint8)
            elif image.shape[2] == 4:
                # RGBA to grayscale
                converted = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                return converted.astype(np.uint8)
        elif target_mode == 'rgb' and len(image.shape) == 2:
            # Convert grayscale to RGB
            return np.stack([image, image, image], axis=-1)
        
        return image
    
    def normalize_image(self, image):
        """
        Normalize image pixel values.
        
        Parameters:
        -----------
        image : numpy array
            Input image
            
        Returns:
        --------
        normalized_image : numpy array
            Normalized image
        """
        if not self.normalize:
            return image
        
        # Convert to float and normalize to [0, 1]
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Already normalized or other format
            return image
    
    def standardize_image(self, image, mean=None, std=None):
        """
        Standardize image using mean and standard deviation.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        mean : float or array, optional
            Mean value(s) for standardization
        std : float or array, optional
            Standard deviation value(s) for standardization
            
        Returns:
        --------
        standardized_image : numpy array
            Standardized image
        """
        if mean is None:
            mean = np.mean(image, axis=(0, 1), keepdims=True)
        if std is None:
            std = np.std(image, axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-8)
        return (image - mean) / std
    
    def apply_gaussian_blur(self, image, sigma=1.0):
        """
        Apply Gaussian blur to reduce noise.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        sigma : float, default=1.0
            Standard deviation for Gaussian kernel
            
        Returns:
        --------
        blurred_image : numpy array
            Blurred image
        """
        if len(image.shape) == 3:
            # Apply blur to each channel separately
            blurred = np.zeros_like(image)
            for i in range(image.shape[2]):
                blurred[:, :, i] = ndimage.gaussian_filter(image[:, :, i], sigma=sigma)
            return blurred
        else:
            # Single channel image
            return ndimage.gaussian_filter(image, sigma=sigma)
    
    def adjust_brightness(self, image, factor=1.0):
        """
        Adjust image brightness.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        factor : float, default=1.0
            Brightness factor (1.0 = no change)
            
        Returns:
        --------
        adjusted_image : numpy array
            Brightness adjusted image
        """
        adjusted = image * factor
        # Clip to valid range
        if image.dtype == np.uint8:
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            adjusted = np.clip(adjusted, 0, 65535).astype(np.uint16)
        return adjusted
    
    def adjust_contrast(self, image, factor=1.0):
        """
        Adjust image contrast.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        factor : float, default=1.0
            Contrast factor (1.0 = no change)
            
        Returns:
        --------
        adjusted_image : numpy array
            Contrast adjusted image
        """
        # Convert to float for processing
        float_image = image.astype(np.float32)
        
        # Calculate mean
        mean = np.mean(float_image)
        
        # Adjust contrast
        adjusted = (float_image - mean) * factor + mean
        
        # Clip to valid range and convert back
        if image.dtype == np.uint8:
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            adjusted = np.clip(adjusted, 0, 65535).astype(np.uint16)
        return adjusted
    
    def rotate_image(self, image, angle):
        """
        Rotate image by specified angle.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        angle : float
            Rotation angle in degrees
            
        Returns:
        --------
        rotated_image : numpy array
            Rotated image
        """
        pil_image = Image.fromarray(image)
        rotated_pil = pil_image.rotate(angle, expand=True)
        return np.array(rotated_pil)
    
    def flip_image(self, image, axis=1):
        """
        Flip image along specified axis.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        axis : int, default=1
            Axis to flip along (0=vertical, 1=horizontal)
            
        Returns:
        --------
        flipped_image : numpy array
            Flipped image
        """
        return np.flip(image, axis=axis)
    
    def preprocess(self, image):
        """
        Apply all preprocessing steps to image.
        
        Parameters:
        -----------
        image : numpy array or str
            Input image array or path to image file
            
        Returns:
        --------
        processed_image : numpy array
            Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Apply preprocessing steps
        if self.target_size:
            image = self.resize_image(image)
        
        image = self.convert_color_space(image)
        image = self.normalize_image(image)
        
        return image


class AdvancedImagePreprocessor:
    """
    Advanced image preprocessing with additional features.
    """
    
    def __init__(self):
        pass
    
    def histogram_equalization(self, image):
        """
        Apply histogram equalization to enhance contrast.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
            
        Returns:
        --------
        equalized_image : numpy array
            Histogram equalized image
        """
        if len(image.shape) == 3:
            raise ValueError("Histogram equalization requires grayscale image")
        
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        
        # Calculate cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        
        # Apply histogram equalization
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        
        return cdf[image]
    
    def adaptive_histogram_equalization(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply adaptive histogram equalization (CLAHE).
        
        Parameters:
        -----------
        image : numpy array
            Input image
        clip_limit : float, default=2.0
            Clipping limit for contrast limiting
        tile_grid_size : tuple, default=(8, 8)
            Size of grid for histogram equalization
            
        Returns:
        --------
        enhanced_image : numpy array
            Enhanced image
        """
        # This is a simplified implementation
        # In practice, you would use cv2.createCLAHE()
        return self.histogram_equalization(image)  # Placeholder
    
    def noise_reduction(self, image, method='gaussian', **kwargs):
        """
        Apply noise reduction to image.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        method : str, default='gaussian'
            Noise reduction method ('gaussian', 'median', 'bilateral')
        **kwargs : additional arguments for specific methods
            
        Returns:
        --------
        denoised_image : numpy array
            Denoised image
        """
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return ndimage.gaussian_filter(image, sigma=sigma)
        elif method == 'median':
            size = kwargs.get('size', 3)
            if len(image.shape) == 3:
                denoised = np.zeros_like(image)
                for i in range(image.shape[2]):
                    denoised[:, :, i] = ndimage.median_filter(image[:, :, i], size=size)
                return denoised
            else:
                return ndimage.median_filter(image, size=size)
        else:
            # Simplified bilateral filter (placeholder)
            return image


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample image for demonstration
    print("Image Preprocessing Demonstration")
    print("=" * 50)
    
    # Create a sample image (checkerboard pattern)
    sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
    sample_image[::10, ::10] = [255, 255, 255]  # White dots
    sample_image[5::10, 5::10] = [255, 0, 0]    # Red dots
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image dtype: {sample_image.dtype}")
    print(f"Sample image value range: [{sample_image.min()}, {sample_image.max()}]")
    
    # Basic preprocessing
    print("\n1. Basic Preprocessing:")
    preprocessor = ImagePreprocessor(target_size=(64, 64), normalize=True)
    processed = preprocessor.preprocess(sample_image)
    
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image dtype: {processed.dtype}")
    print(f"Processed image value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Color space conversion
    print("\n2. Color Space Conversion:")
    gray_preprocessor = ImagePreprocessor(color_mode='grayscale')
    grayscale = gray_preprocessor.convert_color_space(sample_image)
    print(f"Grayscale image shape: {grayscale.shape}")
    
    # Resize image
    print("\n3. Image Resizing:")
    resized = preprocessor.resize_image(sample_image, size=(32, 32))
    print(f"Resized image shape: {resized.shape}")
    
    # Image enhancement
    print("\n4. Image Enhancement:")
    advanced_preprocessor = AdvancedImagePreprocessor()
    
    # Apply Gaussian blur
    blurred = preprocessor.apply_gaussian_blur(sample_image, sigma=1.5)
    print(f"Blurred image shape: {blurred.shape}")
    
    # Adjust brightness
    brightened = preprocessor.adjust_brightness(sample_image, factor=1.5)
    print(f"Brightened image max value: {brightened.max()}")
    
    # Adjust contrast
    contrasted = preprocessor.adjust_contrast(sample_image, factor=1.5)
    print(f"Contrasted image std: {contrasted.std():.2f}")
    
    # Geometric transformations
    print("\n5. Geometric Transformations:")
    
    # Rotate image
    rotated = preprocessor.rotate_image(sample_image, angle=45)
    print(f"Rotated image shape: {rotated.shape}")
    
    # Flip image
    flipped = preprocessor.flip_image(sample_image, axis=1)
    print(f"Flipped image shape: {flipped.shape}")
    
    # Batch preprocessing
    print("\n6. Batch Preprocessing:")
    batch_images = [sample_image, sample_image, sample_image]
    batch_processed = [preprocessor.preprocess(img) for img in batch_images]
    print(f"Batch processed {len(batch_processed)} images")
    print(f"Each image shape: {batch_processed[0].shape}")
    
    # Standardization
    print("\n7. Image Standardization:")
    standardized = preprocessor.standardize_image(processed)
    print(f"Standardized image mean: {standardized.mean():.4f}")
    print(f"Standardized image std: {standardized.std():.4f}")
    
    # Performance comparison
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50)
    
    import time
    
    # Time basic preprocessing
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor.preprocess(sample_image)
    basic_time = time.time() - start_time
    
    # Time advanced preprocessing
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor.apply_gaussian_blur(sample_image, sigma=1.0)
    advanced_time = time.time() - start_time
    
    print(f"Basic preprocessing time (100 iterations): {basic_time:.4f} seconds")
    print(f"Advanced preprocessing time (100 iterations): {advanced_time:.4f} seconds")
    
    # Common preprocessing challenges
    print("\n" + "="*50)
    print("Common Preprocessing Challenges")
    print("="*50)
    print("1. Handling Different Image Formats:")
    print("   - JPEG, PNG, BMP, TIFF support")
    print("   - Color depth variations (8-bit, 16-bit)")
    print("   - Channel order (RGB vs BGR)")
    
    print("\n2. Memory Management:")
    print("   - Large image datasets")
    print("   - Batch processing optimization")
    print("   - GPU memory considerations")
    
    print("\n3. Data Augmentation:")
    print("   - Random transformations")
    print("   - Consistent preprocessing pipelines")
    print("   - Deterministic vs random operations")
    
    print("\n4. Quality Preservation:")
    print("   - Avoiding information loss")
    print("   - Maintaining aspect ratios")
    print("   - Interpolation method selection")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Image Preprocessing")
    print("="*50)
    print("1. Consistency:")
    print("   - Apply same preprocessing to training and inference")
    print("   - Document all preprocessing steps")
    print("   - Version control preprocessing pipelines")
    
    print("\n2. Efficiency:")
    print("   - Use optimized libraries (PIL, OpenCV)")
    print("   - Batch processing for large datasets")
    print("   - Parallel processing when possible")
    
    print("\n3. Quality:")
    print("   - Preserve important image features")
    print("   - Avoid excessive preprocessing")
    print("   - Validate preprocessing results")
    
    print("\n4. Standardization:")
    print("   - Normalize to consistent ranges")
    print("   - Handle different image sizes appropriately")
    print("   - Consider domain-specific requirements")
    
    print("\n5. Augmentation:")
    print("   - Apply data augmentation carefully")
    print("   - Balance augmentation with real data")
    print("   - Monitor for over-augmentation")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- PIL/Pillow: Image loading and basic operations")
    print("- OpenCV: Advanced image processing")
    print("- scikit-image: Scientific image processing")
    print("- TensorFlow/PyTorch: GPU-accelerated preprocessing")
    print("- These provide optimized implementations and GPU support")