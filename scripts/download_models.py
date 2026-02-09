"""Download pre-trained models"""
import os
import torch
from torchvision import models

MODEL_DIR = "models"

def download_pretrained():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Downloading EfficientNet-B4...")
    efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    torch.save(efficientnet.state_dict(), os.path.join(MODEL_DIR, "efficientnet_b4_pretrained.pth"))
    
    print("Downloading ResNet-50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    torch.save(resnet.state_dict(), os.path.join(MODEL_DIR, "resnet50_pretrained.pth"))
    
    print("Done! Models saved to:", MODEL_DIR)

if __name__ == "__main__":
    download_pretrained()
