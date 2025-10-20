"""
Feature Extraction for Computer Vision
====================================

This module demonstrates various feature extraction techniques for computer vision tasks.
It covers edge detection, corner detection, and descriptor methods like SIFT and HOG.

Key Concepts:
- Edge Detection Algorithms
- Corner Detection Methods
- Local Feature Descriptors
- Global Feature Extraction
- Scale-Invariant Features
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import sobel, gaussian_filter
import matplotlib.pyplot as plt


class EdgeDetector:
    """
    Edge detection algorithms for computer vision.
    
    Parameters:
    -----------
    method : str, default='canny'
        Edge detection method ('sobel', 'canny', 'laplacian')
    """
    
    def __init__(self, method='canny'):
        self.method = method
    
    def sobel_edge_detection(self, image, threshold=50):
        """
        Apply Sobel edge detection.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        threshold : int, default=50
            Threshold for edge detection
            
        Returns:
        --------
        edges : numpy array
            Binary edge map
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Apply Sobel operators
        dx = sobel(image, axis=1)
        dy = sobel(image, axis=0)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Apply threshold
        edges = magnitude > threshold
        return edges.astype(np.uint8) * 255
    
    def laplacian_edge_detection(self, image, threshold=30):
        """
        Apply Laplacian edge detection.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        threshold : int, default=30
            Threshold for edge detection
            
        Returns:
        --------
        edges : numpy array
            Binary edge map
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Apply Laplacian operator
        laplacian = ndimage.laplace(image)
        
        # Apply threshold
        edges = np.abs(laplacian) > threshold
        return edges.astype(np.uint8) * 255
    
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """
        Apply Canny edge detection (simplified implementation).
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        low_threshold : int, default=50
            Low threshold for hysteresis
        high_threshold : int, default=150
            High threshold for hysteresis
            
        Returns:
        --------
        edges : numpy array
            Binary edge map
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Apply Gaussian blur to reduce noise
        blurred = gaussian_filter(image, sigma=1.0)
        
        # Calculate gradients using Sobel
        dx = sobel(blurred, axis=1)
        dy = sobel(blurred, axis=0)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        # Non-maximum suppression (simplified)
        suppressed = self.non_maximum_suppression(magnitude, direction)
        
        # Double threshold and edge tracking (simplified)
        strong_edges = suppressed > high_threshold
        weak_edges = (suppressed > low_threshold) & (suppressed <= high_threshold)
        
        # Edge linking (simplified)
        edges = strong_edges | weak_edges
        return edges.astype(np.uint8) * 255
    
    def non_maximum_suppression(self, magnitude, direction):
        """
        Apply non-maximum suppression to gradient magnitude.
        
        Parameters:
        -----------
        magnitude : numpy array
            Gradient magnitude
        direction : numpy array
            Gradient direction
            
        Returns:
        --------
        suppressed : numpy array
            Suppressed gradient magnitude
        """
        rows, cols = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        # Convert angles to degrees
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Get neighboring pixels based on gradient direction
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    # Horizontal edge
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= angle[i,j] < 67.5:
                    # Diagonal edge (45 degrees)
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= angle[i,j] < 112.5:
                    # Vertical edge
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 112.5 <= angle[i,j] < 157.5
                    # Diagonal edge (135 degrees)
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                
                # Suppress if not local maximum
                if magnitude[i,j] >= max(neighbors):
                    suppressed[i,j] = magnitude[i,j]
        
        return suppressed
    
    def detect_edges(self, image, **kwargs):
        """
        Detect edges using specified method.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        **kwargs : additional arguments for specific methods
            
        Returns:
        --------
        edges : numpy array
            Binary edge map
        """
        if self.method == 'sobel':
            threshold = kwargs.get('threshold', 50)
            return self.sobel_edge_detection(image, threshold)
        elif self.method == 'laplacian':
            threshold = kwargs.get('threshold', 30)
            return self.laplacian_edge_detection(image, threshold)
        elif self.method == 'canny':
            low_threshold = kwargs.get('low_threshold', 50)
            high_threshold = kwargs.get('high_threshold', 150)
            return self.canny_edge_detection(image, low_threshold, high_threshold)
        else:
            raise ValueError(f"Unsupported edge detection method: {self.method}")


class CornerDetector:
    """
    Corner detection algorithms for computer vision.
    
    Parameters:
    -----------
    method : str, default='harris'
        Corner detection method ('harris', 'shi-tomasi')
    """
    
    def __init__(self, method='harris'):
        self.method = method
    
    def harris_corner_detection(self, image, k=0.04, threshold=0.01):
        """
        Apply Harris corner detection.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        k : float, default=0.04
            Harris corner detection parameter
        threshold : float, default=0.01
            Threshold for corner response
            
        Returns:
        --------
        corners : numpy array
            Corner response map
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Calculate gradients
        dx = sobel(image, axis=1)
        dy = sobel(image, axis=0)
        
        # Calculate products of derivatives
        Ixx = dx**2
        Iyy = dy**2
        Ixy = dx * dy
        
        # Apply Gaussian filter to products
        Ixx = gaussian_filter(Ixx, sigma=1.0)
        Iyy = gaussian_filter(Iyy, sigma=1.0)
        Ixy = gaussian_filter(Ixy, sigma=1.0)
        
        # Calculate corner response
        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy
        response = det - k * (trace**2)
        
        # Apply threshold
        corners = response > threshold * response.max()
        return corners.astype(np.uint8) * 255
    
    def shi_tomasi_corner_detection(self, image, threshold=0.01):
        """
        Apply Shi-Tomasi corner detection.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        threshold : float, default=0.01
            Threshold for corner response
            
        Returns:
        --------
        corners : numpy array
            Corner response map
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Calculate gradients
        dx = sobel(image, axis=1)
        dy = sobel(image, axis=0)
        
        # Calculate products of derivatives
        Ixx = dx**2
        Iyy = dy**2
        Ixy = dx * dy
        
        # Apply Gaussian filter to products
        Ixx = gaussian_filter(Ixx, sigma=1.0)
        Iyy = gaussian_filter(Iyy, sigma=1.0)
        Ixy = gaussian_filter(Ixy, sigma=1.0)
        
        # Calculate eigenvalues
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        discriminant = trace**2 - 4 * det
        
        # Avoid negative discriminant
        discriminant = np.maximum(discriminant, 0)
        
        # Calculate eigenvalues
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
        
        # Shi-Tomasi response (minimum eigenvalue)
        response = np.minimum(lambda1, lambda2)
        
        # Apply threshold
        corners = response > threshold * response.max()
        return corners.astype(np.uint8) * 255
    
    def detect_corners(self, image, **kwargs):
        """
        Detect corners using specified method.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        **kwargs : additional arguments for specific methods
            
        Returns:
        --------
        corners : numpy array
            Corner response map
        """
        if self.method == 'harris':
            k = kwargs.get('k', 0.04)
            threshold = kwargs.get('threshold', 0.01)
            return self.harris_corner_detection(image, k, threshold)
        elif self.method == 'shi-tomasi':
            threshold = kwargs.get('threshold', 0.01)
            return self.shi_tomasi_corner_detection(image, threshold)
        else:
            raise ValueError(f"Unsupported corner detection method: {self.method}")


class FeatureDescriptor:
    """
    Feature descriptor algorithms for computer vision.
    
    Parameters:
    -----------
    method : str, default='hog'
        Feature descriptor method ('hog', 'sift')
    """
    
    def __init__(self, method='hog'):
        self.method = method
    
    def hog_descriptor(self, image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Compute Histogram of Oriented Gradients (HOG) descriptor.
        
        Parameters:
        -----------
        image : numpy array
            Input image (grayscale)
        orientations : int, default=9
            Number of orientation bins
        pixels_per_cell : tuple, default=(8, 8)
            Size of cell in pixels
        cells_per_block : tuple, default=(2, 2)
            Number of cells in each block
            
        Returns:
        --------
        features : numpy array
            HOG feature vector
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Calculate gradients
        dx = sobel(image, axis=1)
        dy = sobel(image, axis=0)
        
        # Calculate gradient magnitude and orientation
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.rad2deg(np.arctan2(dy, dx)) % 180
        
        # Calculate cell dimensions
        cell_h, cell_w = pixels_per_cell
        block_h, block_w = cells_per_block
        rows, cols = image.shape
        
        # Calculate number of cells
        n_cells_y = rows // cell_h
        n_cells_x = cols // cell_w
        
        # Initialize HOG features
        hog_features = []
        
        # Process each cell
        for i in range(n_cells_y):
            for j in range(n_cells_x):
                # Extract cell region
                cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_ori = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                
                # Calculate histogram for cell
                hist, _ = np.histogram(cell_ori, bins=orientations, 
                                     range=(0, 180), weights=cell_mag)
                hog_features.extend(hist)
        
        # Normalize features (simplified)
        hog_features = np.array(hog_features)
        if np.linalg.norm(hog_features) > 0:
            hog_features = hog_features / np.linalg.norm(hog_features)
        
        return hog_features
    
    def extract_features(self, image, **kwargs):
        """
        Extract features using specified method.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        **kwargs : additional arguments for specific methods
            
        Returns:
        --------
        features : numpy array
            Feature vector
        """
        if self.method == 'hog':
            orientations = kwargs.get('orientations', 9)
            pixels_per_cell = kwargs.get('pixels_per_cell', (8, 8))
            cells_per_block = kwargs.get('cells_per_block', (2, 2))
            return self.hog_descriptor(image, orientations, pixels_per_cell, cells_per_block)
        else:
            # Placeholder for other methods (SIFT, etc.)
            return np.array([])


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample image for demonstration
    print("Feature Extraction for Computer Vision Demonstration")
    print("=" * 55)
    
    # Create a sample image with geometric shapes
    sample_image = np.zeros((100, 100), dtype=np.uint8)
    # Draw a square
    sample_image[20:80, 20:80] = 255
    # Draw a circle
    y, x = np.ogrid[:100, :100]
    mask = (x - 50)**2 + (y - 50)**2 <= 25**2
    sample_image[mask] = 128
    
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image dtype: {sample_image.dtype}")
    print(f"Sample image value range: [{sample_image.min()}, {sample_image.max()}]")
    
    # Edge Detection
    print("\n1. Edge Detection:")
    
    # Sobel edge detection
    edge_detector = EdgeDetector(method='sobel')
    sobel_edges = edge_detector.detect_edges(sample_image, threshold=50)
    print(f"Sobel edges shape: {sobel_edges.shape}")
    print(f"Sobel edges non-zero pixels: {np.count_nonzero(sobel_edges)}")
    
    # Canny edge detection
    canny_detector = EdgeDetector(method='canny')
    canny_edges = canny_detector.detect_edges(sample_image, low_threshold=30, high_threshold=100)
    print(f"Canny edges shape: {canny_edges.shape}")
    print(f"Canny edges non-zero pixels: {np.count_nonzero(canny_edges)}")
    
    # Laplacian edge detection
    laplacian_detector = EdgeDetector(method='laplacian')
    laplacian_edges = laplacian_detector.detect_edges(sample_image, threshold=20)
    print(f"Laplacian edges shape: {laplacian_edges.shape}")
    print(f"Laplacian edges non-zero pixels: {np.count_nonzero(laplacian_edges)}")
    
    # Corner Detection
    print("\n2. Corner Detection:")
    
    # Harris corner detection
    corner_detector = CornerDetector(method='harris')
    harris_corners = corner_detector.detect_corners(sample_image, k=0.04, threshold=0.01)
    print(f"Harris corners shape: {harris_corners.shape}")
    print(f"Harris corners non-zero pixels: {np.count_nonzero(harris_corners)}")
    
    # Shi-Tomasi corner detection
    shi_tomasi_detector = CornerDetector(method='shi-tomasi')
    shi_tomasi_corners = shi_tomasi_detector.detect_corners(sample_image, threshold=0.01)
    print(f"Shi-Tomasi corners shape: {shi_tomasi_corners.shape}")
    print(f"Shi-Tomasi corners non-zero pixels: {np.count_nonzero(shi_tomasi_corners)}")
    
    # Feature Descriptors
    print("\n3. Feature Descriptors:")
    
    # HOG descriptor
    feature_descriptor = FeatureDescriptor(method='hog')
    hog_features = feature_descriptor.extract_features(sample_image, 
                                                     orientations=9,
                                                     pixels_per_cell=(8, 8),
                                                     cells_per_block=(2, 2))
    print(f"HOG features shape: {hog_features.shape}")
    print(f"HOG features L2 norm: {np.linalg.norm(hog_features):.4f}")
    
    # Compare different methods
    print("\n" + "="*50)
    print("Comparison of Feature Extraction Methods")
    print("="*50)
    
    # Create a more complex sample image
    complex_image = np.zeros((128, 128), dtype=np.uint8)
    # Add multiple shapes
    complex_image[20:40, 20:40] = 255  # Square
    complex_image[60:80, 60:80] = 200  # Another square
    complex_image[100:120, 20:40] = 150  # Third square
    
    # Time different edge detection methods
    import time
    
    methods = ['sobel', 'canny', 'laplacian']
    edge_times = {}
    
    for method in methods:
        detector = EdgeDetector(method=method)
        start_time = time.time()
        edges = detector.detect_edges(complex_image)
        edge_times[method] = time.time() - start_time
        print(f"{method.capitalize()} edge detection time: {edge_times[method]:.4f} seconds")
    
    # Compare corner detection methods
    corner_methods = ['harris', 'shi-tomasi']
    corner_times = {}
    
    for method in corner_methods:
        detector = CornerDetector(method=method)
        start_time = time.time()
        corners = detector.detect_corners(complex_image)
        corner_times[method] = time.time() - start_time
        print(f"{method.capitalize()} corner detection time: {corner_times[method]:.4f} seconds")
    
    # Feature descriptor comparison
    print("\nFeature Descriptor Performance:")
    start_time = time.time()
    hog_features = feature_descriptor.extract_features(complex_image)
    hog_time = time.time() - start_time
    print(f"HOG descriptor extraction time: {hog_time:.4f} seconds")
    print(f"HOG feature vector dimension: {len(hog_features)}")
    
    # Advanced feature extraction concepts
    print("\n" + "="*50)
    print("Advanced Feature Extraction Concepts")
    print("="*50)
    print("1. Scale-Invariant Features:")
    print("   - SIFT (Scale-Invariant Feature Transform)")
    print("   - SURF (Speeded-Up Robust Features)")
    print("   - Handle scale and rotation variations")
    
    print("\n2. Local Binary Patterns (LBP):")
    print("   - Texture descriptor")
    print("   - Robust to illumination changes")
    print("   - Efficient computation")
    
    print("\n3. Deep Learning Features:")
    print("   - CNN-based feature extraction")
    print("   - Transfer learning with pre-trained models")
    print("   - End-to-end feature learning")
    
    print("\n4. Multi-scale Features:")
    print("   - Pyramid representations")
    print("   - Multi-resolution analysis")
    print("   - Scale-space theory")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of Feature Extraction")
    print("="*50)
    print("1. Object Detection:")
    print("   - HOG + SVM for pedestrian detection")
    print("   - SIFT for object recognition")
    print("   - Deep features for modern detectors")
    
    print("\n2. Image Matching:")
    print("   - Feature matching between images")
    print("   - Homography estimation")
    print("   - Image stitching and panorama creation")
    
    print("\n3. Motion Analysis:")
    print("   - Optical flow computation")
    print("   - Feature tracking in video sequences")
    print("   - Motion estimation and compensation")
    
    print("\n4. Medical Imaging:")
    print("   - Edge detection for organ segmentation")
    print("   - Texture analysis for disease diagnosis")
    print("   - Feature extraction for CAD systems")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Feature Extraction")
    print("="*50)
    print("1. Method Selection:")
    print("   - Choose appropriate methods for your task")
    print("   - Consider computational requirements")
    print("   - Balance accuracy and efficiency")
    
    print("\n2. Parameter Tuning:")
    print("   - Experiment with different parameters")
    print("   - Validate on representative data")
    print("   - Use cross-validation for robust evaluation")
    
    print("\n3. Preprocessing:")
    print("   - Apply appropriate image preprocessing")
    print("   - Handle noise and artifacts")
    print("   - Normalize input data consistently")
    
    print("\n4. Evaluation:")
    print("   - Use task-specific evaluation metrics")
    print("   - Compare with baseline methods")
    print("   - Validate on independent test sets")
    
    print("\n5. Integration:")
    print("   - Combine multiple feature types")
    print("   - Use ensemble approaches")
    print("   - Consider end-to-end learning")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- OpenCV: Comprehensive computer vision library")
    print("- scikit-image: Scientific image processing")
    print("- TensorFlow/PyTorch: Deep learning features")
    print("- These provide optimized implementations and GPU support")