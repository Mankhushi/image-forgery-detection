"""Data Augmentation Module"""
import cv2
import numpy as np
from typing import Tuple
import random

class DataAugmentation:
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return cv2.flip(image, 1)
        return image
    
    def vertical_flip(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return cv2.flip(image, 0)
        return image
    
    def rotate(self, image: np.ndarray, angle: int = 90) -> np.ndarray:
        if random.random() < self.p:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            return cv2.warpAffine(image, M, (w, h))
        return image
    
    def brightness(self, image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        if random.random() < self.p:
            f = 1 + random.uniform(-factor, factor)
            return np.clip(image * f, 0, 255).astype(np.uint8)
        return image
    
    def noise(self, image: np.ndarray, std: float = 10) -> np.ndarray:
        if random.random() < self.p:
            noise = np.random.normal(0, std, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
        return image
    
    def apply_all(self, image: np.ndarray) -> np.ndarray:
        image = self.horizontal_flip(image)
        image = self.vertical_flip(image)
        image = self.rotate(image)
        image = self.brightness(image)
        return image
