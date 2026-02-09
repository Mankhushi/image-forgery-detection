"""
Image Preprocessing Module
Handles image loading, resizing, normalization, and format conversion
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Union, Optional
import io


class ImageProcessor:
    """
    Production-grade image processor for forgery detection
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # PyTorch transforms
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transform_list)
        
        # Inverse transform for visualization
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-m/s for m, s in zip(mean, std)],
                std=[1/s for s in std]
            )
        ])
    
    def load_image(
        self,
        image_source: Union[str, bytes, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Load image from various sources
        
        Args:
            image_source: File path, bytes, numpy array, or PIL Image
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_source, str):
            # File path
            image = Image.open(image_source)
        elif isinstance(image_source, bytes):
            # Bytes data
            image = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, np.ndarray):
            # Numpy array (BGR from OpenCV)
            if len(image_source.shape) == 3 and image_source.shape[2] == 3:
                image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_source)
        elif isinstance(image_source, Image.Image):
            image = image_source
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def preprocess(
        self,
        image: Union[str, bytes, np.ndarray, Image.Image],
        return_original: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Image.Image]]:
        """
        Preprocess image for model input
        
        Args:
            image: Input image
            return_original: Whether to return original image
            
        Returns:
            Preprocessed tensor, optionally with original image
        """
        pil_image = self.load_image(image)
        tensor = self.transform(pil_image)
        
        if return_original:
            return tensor, pil_image
        return tensor
    
    def preprocess_batch(
        self,
        images: list,
        return_original: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            return_original: Whether to return original images
            
        Returns:
            Batch tensor, optionally with original images
        """
        tensors = []
        originals = []
        
        for img in images:
            if return_original:
                tensor, original = self.preprocess(img, return_original=True)
                originals.append(original)
            else:
                tensor = self.preprocess(img)
            tensors.append(tensor)
        
        batch_tensor = torch.stack(tensors)
        
        if return_original:
            return batch_tensor, originals
        return batch_tensor
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array for visualization
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array (H, W, C) in range [0, 255]
        """
        if self.normalize:
            tensor = self.inverse_transform(tensor)
        
        # Clamp values
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        numpy_img = tensor.permute(1, 2, 0).cpu().numpy()
        numpy_img = (numpy_img * 255).astype(np.uint8)
        
        return numpy_img
    
    def resize_image(
        self,
        image: Union[np.ndarray, Image.Image],
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            size: Target size (width, height)
            
        Returns:
            Resized numpy array
        """
        if size is None:
            size = self.target_size
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return resized
    
    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: int = 64,
        stride: int = 32
    ) -> np.ndarray:
        """
        Extract overlapping patches from image
        
        Args:
            image: Input image (H, W, C)
            patch_size: Size of each patch
            stride: Stride between patches
            
        Returns:
            Array of patches (N, patch_size, patch_size, C)
        """
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
        
        return np.array(patches)
    
    def get_image_metadata(self, image_path: str) -> dict:
        """
        Extract image metadata
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with metadata
        """
        image = Image.open(image_path)
        
        metadata = {
            'format': image.format,
            'mode': image.mode,
            'size': image.size,
            'width': image.width,
            'height': image.height,
        }
        
        # EXIF data
        if hasattr(image, '_getexif') and image._getexif():
            metadata['exif'] = dict(image._getexif())
        
        return metadata
