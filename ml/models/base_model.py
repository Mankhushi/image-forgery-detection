"""Base Model Class for all detectors"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np

class BaseDetector(ABC, nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def predict(self, image: torch.Tensor) -> Dict:
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self.forward(image)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = probs[0, pred_class[0]].item()
        
        return {
            'prediction': 'forged' if pred_class.item() == 1 else 'authentic',
            'confidence': confidence,
            'probabilities': {'authentic': probs[0, 0].item(), 'forged': probs[0, 1].item()}
        }
    
    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
    
    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)
