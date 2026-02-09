"""Copy-Move Forgery Detector using EfficientNet"""
import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseDetector

class CopyMoveDetector(BaseDetector):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__(num_classes, pretrained)
        
        # EfficientNet-B4 backbone
        if pretrained:
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b4(weights=None)
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier"""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        return features.flatten(1)
