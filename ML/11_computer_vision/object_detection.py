"""
Object Detection Implementation
=============================

This module demonstrates object detection algorithms including YOLO, R-CNN, and SSD.
It covers bounding box representations, non-maximum suppression, and evaluation metrics.

Key Concepts:
- Bounding Box Representations
- Intersection over Union (IoU)
- Non-Maximum Suppression
- Object Detection Architectures
- Evaluation Metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class BoundingBox:
    """
    Bounding box representation and operations.
    
    Parameters:
    -----------
    x : float
        X coordinate of top-left corner
    y : float
        Y coordinate of top-left corner
    width : float
        Width of bounding box
    height : float
        Height of bounding box
    confidence : float, default=1.0
        Confidence score
    class_id : int, default=0
        Class identifier
    """
    
    def __init__(self, x, y, width, height, confidence=1.0, class_id=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_id = class_id
    
    def area(self):
        """Calculate area of bounding box."""
        return self.width * self.height
    
    def intersection(self, other):
        """
        Calculate intersection area with another bounding box.
        
        Parameters:
        -----------
        other : BoundingBox
            Other bounding box
            
        Returns:
        --------
        intersection_area : float
            Intersection area
        """
        # Calculate intersection coordinates
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x + self.width, other.x + other.width)
        y_bottom = min(self.y + self.height, other.y + other.height)
        
        # Check if there is an intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        # Calculate intersection area
        return (x_right - x_left) * (y_bottom - y_top)
    
    def union(self, other):
        """
        Calculate union area with another bounding box.
        
        Parameters:
        -----------
        other : BoundingBox
            Other bounding box
            
        Returns:
        --------
        union_area : float
            Union area
        """
        return self.area() + other.area() - self.intersection(other)
    
    def iou(self, other):
        """
        Calculate Intersection over Union (IoU) with another bounding box.
        
        Parameters:
        -----------
        other : BoundingBox
            Other bounding box
            
        Returns:
        --------
        iou : float
            IoU value
        """
        intersection_area = self.intersection(other)
        union_area = self.union(other)
        
        # Avoid division by zero
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def to_xyxy(self):
        """
        Convert to (x1, y1, x2, y2) format.
        
        Returns:
        --------
        coordinates : tuple
            (x1, y1, x2, y2) coordinates
        """
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def to_cxcywh(self):
        """
        Convert to (center_x, center_y, width, height) format.
        
        Returns:
        --------
        coordinates : tuple
            (center_x, center_y, width, height) coordinates
        """
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        return (center_x, center_y, self.width, self.height)


def non_maximum_suppression(boxes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove redundant bounding boxes.
    
    Parameters:
    -----------
    boxes : list
        List of BoundingBox objects
    iou_threshold : float, default=0.5
        IoU threshold for suppression
        
    Returns:
    --------
    filtered_boxes : list
        List of filtered BoundingBox objects
    """
    if len(boxes) == 0:
        return []
    
    # Sort boxes by confidence (descending)
    sorted_boxes = sorted(boxes, key=lambda x: x.confidence, reverse=True)
    
    # Keep track of boxes to suppress
    suppressed = [False] * len(sorted_boxes)
    filtered_boxes = []
    
    for i in range(len(sorted_boxes)):
        if suppressed[i]:
            continue
            
        # Keep this box
        current_box = sorted_boxes[i]
        filtered_boxes.append(current_box)
        
        # Suppress boxes with high IoU
        for j in range(i + 1, len(sorted_boxes)):
            if suppressed[j]:
                continue
                
            iou = current_box.iou(sorted_boxes[j])
            if iou > iou_threshold:
                suppressed[j] = True
    
    return filtered_boxes


class YOLODetector:
    """
    Simplified YOLO (You Only Look Once) object detector.
    
    Parameters:
    -----------
    input_size : tuple, default=(416, 416)
        Input image size (height, width)
    num_classes : int, default=80
        Number of object classes
    anchors : list, optional
        Anchor boxes
    """
    
    def __init__(self, input_size=(416, 416), num_classes=80, anchors=None):
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Default anchors for COCO dataset
        if anchors is None:
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # Small objects
                [(30, 61), (62, 45), (59, 119)],     # Medium objects
                [(116, 90), (156, 198), (373, 326)]  # Large objects
            ]
        else:
            self.anchors = anchors
    
    def detect(self, image):
        """
        Perform object detection on image.
        
        Parameters:
        -----------
        image : numpy array
            Input image
            
        Returns:
        --------
        detections : list
            List of detected bounding boxes
        """
        # In a real implementation, this would involve:
        # 1. Preprocessing the image
        # 2. Forward pass through the network
        # 3. Post-processing detections
        # 4. Applying NMS
        
        # For demonstration, we'll generate sample detections
        height, width = image.shape[:2]
        
        # Generate sample detections
        detections = []
        num_detections = np.random.randint(3, 8)
        
        for _ in range(num_detections):
            # Random bounding box
            x = np.random.randint(0, width - 50)
            y = np.random.randint(0, height - 50)
            box_width = np.random.randint(30, min(100, width - x))
            box_height = np.random.randint(30, min(100, height - y))
            confidence = np.random.uniform(0.5, 1.0)
            class_id = np.random.randint(0, self.num_classes)
            
            detections.append(BoundingBox(x, y, box_width, box_height, confidence, class_id))
        
        # Apply non-maximum suppression
        filtered_detections = non_maximum_suppression(detections, iou_threshold=0.5)
        
        return filtered_detections


class RCNNDetector:
    """
    Simplified R-CNN (Regions with CNN features) object detector.
    
    Parameters:
    -----------
    num_classes : int, default=80
        Number of object classes
    """
    
    def __init__(self, num_classes=80):
        self.num_classes = num_classes
    
    def generate_proposals(self, image):
        """
        Generate region proposals using selective search or other methods.
        
        Parameters:
        -----------
        image : numpy array
            Input image
            
        Returns:
        --------
        proposals : list
            List of proposed bounding boxes
        """
        # In a real implementation, this would use selective search or RPN
        # For demonstration, we'll generate sample proposals
        height, width = image.shape[:2]
        
        proposals = []
        num_proposals = np.random.randint(10, 20)
        
        for _ in range(num_proposals):
            # Random bounding box
            x = np.random.randint(0, width - 30)
            y = np.random.randint(0, height - 30)
            box_width = np.random.randint(20, min(80, width - x))
            box_height = np.random.randint(20, min(80, height - y))
            
            proposals.append(BoundingBox(x, y, box_width, box_height))
        
        return proposals
    
    def classify_proposals(self, image, proposals):
        """
        Classify region proposals using CNN features.
        
        Parameters:
        -----------
        image : numpy array
            Input image
        proposals : list
            List of bounding box proposals
            
        Returns:
        --------
        detections : list
            List of classified detections
        """
        # In a real implementation, this would:
        # 1. Extract features from each proposal
        # 2. Classify each proposal
        # 3. Refine bounding box coordinates
        
        # For demonstration, we'll add confidence scores and class IDs
        detections = []
        
        for proposal in proposals:
            # Random confidence and class
            confidence = np.random.uniform(0.3, 0.9)
            class_id = np.random.randint(0, self.num_classes)
            
            # Create detection with confidence and class
            detection = BoundingBox(
                proposal.x, proposal.y, proposal.width, proposal.height,
                confidence, class_id
            )
            detections.append(detection)
        
        # Apply non-maximum suppression
        filtered_detections = non_maximum_suppression(detections, iou_threshold=0.7)
        
        return filtered_detections
    
    def detect(self, image):
        """
        Perform object detection using R-CNN approach.
        
        Parameters:
        -----------
        image : numpy array
            Input image
            
        Returns:
        --------
        detections : list
            List of detected bounding boxes
        """
        # Generate region proposals
        proposals = self.generate_proposals(image)
        
        # Classify proposals
        detections = self.classify_proposals(image, proposals)
        
        return detections


class EvaluationMetrics:
    """
    Object detection evaluation metrics.
    """
    
    @staticmethod
    def calculate_precision_recall(detections, ground_truth, iou_threshold=0.5):
        """
        Calculate precision and recall for object detection.
        
        Parameters:
        -----------
        detections : list
            List of detected bounding boxes
        ground_truth : list
            List of ground truth bounding boxes
        iou_threshold : float, default=0.5
            IoU threshold for matching
            
        Returns:
        --------
        precision : float
            Precision value
        recall : float
            Recall value
        """
        if len(detections) == 0:
            return 0.0, 0.0
        
        if len(ground_truth) == 0:
            return 0.0, 0.0 if len(detections) > 0 else 1.0
        
        # Match detections to ground truth
        true_positives = 0
        matched_gt = set()
        
        for detection in detections:
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                    
                iou = detection.iou(gt)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = i
            
            # If matched, count as true positive
            if best_gt_idx != -1:
                true_positives += 1
                matched_gt.add(best_gt_idx)
        
        # Calculate precision and recall
        precision = true_positives / len(detections) if len(detections) > 0 else 0.0
        recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
        
        return precision, recall
    
    @staticmethod
    def calculate_map(detections_list, ground_truth_list, iou_threshold=0.5):
        """
        Calculate mean Average Precision (mAP).
        
        Parameters:
        -----------
        detections_list : list
            List of detection lists for each image
        ground_truth_list : list
            List of ground truth lists for each image
        iou_threshold : float, default=0.5
            IoU threshold for matching
            
        Returns:
        --------
        map_score : float
            Mean Average Precision
        """
        # This is a simplified implementation
        # In practice, mAP calculation is more complex
        
        total_precision = 0
        total_recall = 0
        num_images = len(detections_list)
        
        for i in range(num_images):
            detections = detections_list[i]
            ground_truth = ground_truth_list[i]
            
            precision, recall = EvaluationMetrics.calculate_precision_recall(
                detections, ground_truth, iou_threshold
            )
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / num_images if num_images > 0 else 0.0
        avg_recall = total_recall / num_images if num_images > 0 else 0.0
        
        return avg_precision, avg_recall


# Example usage and demonstration
if __name__ == "__main__":
    # Create sample image for demonstration
    print("Object Detection Implementation Demonstration")
    print("=" * 50)
    
    # Create a sample image (3 channels, 416x416)
    sample_image = np.random.randn(416, 416, 3).astype(np.float32)
    
    print(f"Sample image shape: {sample_image.shape}")
    
    # Bounding Box demonstration
    print("\n1. Bounding Box Operations:")
    
    # Create sample bounding boxes
    box1 = BoundingBox(50, 50, 100, 100, confidence=0.9, class_id=1)
    box2 = BoundingBox(75, 75, 100, 100, confidence=0.8, class_id=1)
    
    print(f"Box 1: x={box1.x}, y={box1.y}, w={box1.width}, h={box1.height}")
    print(f"Box 2: x={box2.x}, y={box2.y}, w={box2.width}, h={box2.height}")
    
    # Calculate IoU
    iou = box1.iou(box2)
    print(f"IoU between boxes: {iou:.3f}")
    
    # Convert formats
    xyxy = box1.to_xyxy()
    cxcywh = box1.to_cxcywh()
    print(f"Box 1 in (x1,y1,x2,y2) format: {xyxy}")
    print(f"Box 1 in (cx,cy,w,h) format: {cxcywh}")
    
    # Non-Maximum Suppression demonstration
    print("\n2. Non-Maximum Suppression:")
    
    # Create overlapping bounding boxes
    overlapping_boxes = [
        BoundingBox(50, 50, 100, 100, confidence=0.9, class_id=1),
        BoundingBox(60, 60, 100, 100, confidence=0.8, class_id=1),
        BoundingBox(70, 70, 100, 100, confidence=0.7, class_id=1),
        BoundingBox(200, 200, 80, 80, confidence=0.95, class_id=2),  # Different class
        BoundingBox(210, 210, 80, 80, confidence=0.85, class_id=2)   # Different class
    ]
    
    print(f"Number of input boxes: {len(overlapping_boxes)}")
    
    # Apply NMS
    filtered_boxes = non_maximum_suppression(overlapping_boxes, iou_threshold=0.5)
    print(f"Number of boxes after NMS: {len(filtered_boxes)}")
    
    # YOLO Detector demonstration
    print("\n3. YOLO Detector:")
    yolo_detector = YOLODetector(input_size=(416, 416), num_classes=10)
    
    # Perform detection
    yolo_detections = yolo_detector.detect(sample_image)
    print(f"YOLO detections: {len(yolo_detections)}")
    
    # Show sample detections
    for i, detection in enumerate(yolo_detections[:3]):
        print(f"  Detection {i+1}: class={detection.class_id}, "
              f"confidence={detection.confidence:.3f}, "
              f"bbox=({detection.x:.0f},{detection.y:.0f},{detection.width:.0f},{detection.height:.0f})")
    
    # R-CNN Detector demonstration
    print("\n4. R-CNN Detector:")
    rcnn_detector = RCNNDetector(num_classes=10)
    
    # Perform detection
    rcnn_detections = rcnn_detector.detect(sample_image)
    print(f"R-CNN detections: {len(rcnn_detections)}")
    
    # Show sample detections
    for i, detection in enumerate(rcnn_detections[:3]):
        print(f"  Detection {i+1}: class={detection.class_id}, "
              f"confidence={detection.confidence:.3f}, "
              f"bbox=({detection.x:.0f},{detection.y:.0f},{detection.width:.0f},{detection.height:.0f})")
    
    # Evaluation Metrics demonstration
    print("\n5. Evaluation Metrics:")
    
    # Create sample ground truth
    ground_truth = [
        BoundingBox(45, 45, 105, 105, confidence=1.0, class_id=1),
        BoundingBox(195, 195, 85, 85, confidence=1.0, class_id=2)
    ]
    
    # Calculate precision and recall
    precision, recall = EvaluationMetrics.calculate_precision_recall(
        yolo_detections, ground_truth, iou_threshold=0.5
    )
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # Calculate mAP (simplified)
    detections_list = [yolo_detections, rcnn_detections]
    ground_truth_list = [ground_truth, ground_truth]
    
    avg_precision, avg_recall = EvaluationMetrics.calculate_map(
        detections_list, ground_truth_list, iou_threshold=0.5
    )
    
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    
    # Compare different object detection architectures
    print("\n" + "="*50)
    print("Comparison of Object Detection Architectures")
    print("="*50)
    
    print("1. YOLO (You Only Look Once):")
    print("   - Single-stage detector")
    print("   - Real-time performance")
    print("   - Divides image into grid cells")
    print("   - Predicts bounding boxes and class probabilities")
    
    print("\n2. R-CNN (Regions with CNN features):")
    print("   - Two-stage detector")
    print("   - Generates region proposals first")
    print("   - Classifies each proposal")
    print("   - High accuracy but slower")
    
    print("\n3. Fast R-CNN:")
    print("   - Improved R-CNN")
    print("   - ROI pooling layer")
    print("   - Single network for classification and regression")
    
    print("\n4. Faster R-CNN:")
    print("   - Adds Region Proposal Network (RPN)")
    print("   - End-to-end training")
    print("   - Better performance than Fast R-CNN")
    
    print("\n5. SSD (Single Shot MultiBox Detector):")
    print("   - Single-stage detector")
    print("   - Multi-scale feature maps")
    print("   - Good balance of speed and accuracy")
    
    print("\n6. RetinaNet:")
    print("   - Addresses class imbalance")
    print("   - Focal loss function")
    print("   - High accuracy with good speed")
    
    # Advanced object detection concepts
    print("\n" + "="*50)
    print("Advanced Object Detection Concepts")
    print("="*50)
    print("1. Anchor Boxes:")
    print("   - Pre-defined bounding box shapes")
    print("   - Handle different aspect ratios")
    print("   - Improve detection accuracy")
    
    print("\n2. Feature Pyramid Networks (FPN):")
    print("   - Multi-scale feature extraction")
    print("   - Combine high-level semantics with low-level details")
    print("   - Improve detection of small objects")
    
    print("\n3. Data Augmentation:")
    print("   - Random cropping and scaling")
    print("   - Color jittering")
    print("   - MixUp and CutMix for object detection")
    
    print("\n4. Model Ensemble:")
    print("   - Combine multiple detectors")
    print("   - Weighted averaging of predictions")
    print("   - Improve robustness and accuracy")
    
    # Applications
    print("\n" + "="*50)
    print("Applications of Object Detection")
    print("="*50)
    print("1. Autonomous Vehicles:")
    print("   - Pedestrian detection")
    print("   - Vehicle detection")
    print("   - Traffic sign recognition")
    
    print("\n2. Surveillance Systems:")
    print("   - Intruder detection")
    print("   - Anomaly detection")
    print("   - Crowd monitoring")
    
    print("\n3. Medical Imaging:")
    print("   - Tumor detection")
    print("   - Organ segmentation")
    print("   - Cell counting")
    
    print("\n4. Retail and E-commerce:")
    print("   - Product detection")
    print("   - Shelf monitoring")
    print("   - Inventory management")
    
    print("\n5. Robotics:")
    print("   - Object manipulation")
    print("   - Navigation and mapping")
    print("   - Human-robot interaction")
    
    # Best practices
    print("\n" + "="*50)
    print("Best Practices for Object Detection")
    print("="*50)
    print("1. Dataset Preparation:")
    print("   - High-quality annotations")
    print("   - Diverse and representative samples")
    print("   - Handle class imbalance appropriately")
    
    print("\n2. Model Selection:")
    print("   - Choose appropriate architecture for requirements")
    print("   - Consider speed vs accuracy trade-offs")
    print("   - Use pre-trained models when possible")
    
    print("\n3. Training Strategy:")
    print("   - Use appropriate learning rates")
    print("   - Implement data augmentation")
    print("   - Monitor for overfitting")
    
    print("\n4. Evaluation:")
    print("   - Use standard metrics (mAP, IoU)")
    print("   - Validate on independent test sets")
    print("   - Consider domain-specific requirements")
    
    print("\n5. Deployment:")
    print("   - Optimize for inference speed")
    print("   - Consider edge deployment constraints")
    print("   - Implement proper error handling")
    
    # Note about production implementations
    print("\n" + "="*50)
    print("Production Implementation Note:")
    print("="*50)
    print("For production use, consider using optimized libraries:")
    print("- TensorFlow Object Detection API")
    print("- PyTorch torchvision models")
    print("- OpenCV DNN module")
    print("- These provide optimized implementations and pre-trained models")