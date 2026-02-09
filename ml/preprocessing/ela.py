"""
Error Level Analysis (ELA) Module
Detects image manipulation by analyzing JPEG compression artifacts
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional


class ELAProcessor:
    """
    Error Level Analysis for forgery detection
    
    ELA works by re-saving the image at a known quality level and
    comparing it to the original. Manipulated regions show different
    error levels than the rest of the image.
    """
    
    def __init__(
        self,
        quality: int = 90,
        scale: int = 15,
        threshold: Optional[int] = None
    ):
        """
        Initialize ELA processor
        
        Args:
            quality: JPEG quality for re-compression (1-100)
            scale: Scale factor for error amplification
            threshold: Optional threshold for binary mask
        """
        self.quality = quality
        self.scale = scale
        self.threshold = threshold
    
    def compute_ela(
        self,
        image: np.ndarray,
        return_difference: bool = False
    ) -> np.ndarray:
        """
        Compute Error Level Analysis
        
        Args:
            image: Input image (RGB numpy array)
            return_difference: Return raw difference instead of scaled
            
        Returns:
            ELA image showing error levels
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Re-save at specified quality
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        
        # Load re-compressed image
        resaved = Image.open(buffer)
        
        # Convert both to numpy arrays
        original = np.array(pil_image).astype(np.float32)
        compressed = np.array(resaved).astype(np.float32)
        
        # Compute absolute difference
        difference = np.abs(original - compressed)
        
        if return_difference:
            return difference
        
        # Scale the difference for visualization
        ela = difference * self.scale
        ela = np.clip(ela, 0, 255).astype(np.uint8)
        
        return ela
    
    def compute_ela_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Compute grayscale ELA (single channel)
        
        Args:
            image: Input image
            
        Returns:
            Grayscale ELA image
        """
        ela = self.compute_ela(image)
        
        # Convert to grayscale
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_RGB2GRAY)
        
        return ela_gray
    
    def get_ela_heatmap(
        self,
        image: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generate ELA heatmap
        
        Args:
            image: Input image
            colormap: OpenCV colormap
            
        Returns:
            Colored heatmap
        """
        ela_gray = self.compute_ela_grayscale(image)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(ela_gray, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def get_suspicious_regions(
        self,
        image: np.ndarray,
        threshold: Optional[int] = None
    ) -> Tuple[np.ndarray, list]:
        """
        Detect suspicious (potentially manipulated) regions
        
        Args:
            image: Input image
            threshold: Threshold for binary mask (auto if None)
            
        Returns:
            Binary mask and list of bounding boxes
        """
        ela_gray = self.compute_ela_grayscale(image)
        
        # Auto threshold using Otsu's method
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            _, binary = cv2.threshold(
                ela_gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(ela_gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get bounding boxes
        bboxes = []
        min_area = 100  # Minimum area to consider
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area)
                })
        
        return binary, bboxes
    
    def overlay_ela(
        self,
        original: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay ELA heatmap on original image
        
        Args:
            original: Original image
            alpha: Blend factor
            
        Returns:
            Blended image
        """
        heatmap = self.get_ela_heatmap(original)
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != original.shape[:2]:
            heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        
        # Blend
        blended = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
        
        return blended
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        Complete ELA analysis
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with analysis results
        """
        ela = self.compute_ela(image)
        ela_gray = self.compute_ela_grayscale(image)
        heatmap = self.get_ela_heatmap(image)
        binary, bboxes = self.get_suspicious_regions(image)
        
        # Compute statistics
        mean_error = np.mean(ela_gray)
        max_error = np.max(ela_gray)
        std_error = np.std(ela_gray)
        
        # Suspicious score (0-1)
        suspicious_score = min(1.0, mean_error / 50.0)
        
        return {
            'ela': ela,
            'ela_grayscale': ela_gray,
            'heatmap': heatmap,
            'binary_mask': binary,
            'suspicious_regions': bboxes,
            'statistics': {
                'mean_error': float(mean_error),
                'max_error': float(max_error),
                'std_error': float(std_error),
                'suspicious_score': float(suspicious_score)
            }
        }
