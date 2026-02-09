"""Dataset classes for training"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Callable

class ForgeryDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, train: bool = True):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        authentic_dir = os.path.join(root_dir, 'authentic')
        forged_dir = os.path.join(root_dir, 'forged')
        
        if os.path.exists(authentic_dir):
            for f in os.listdir(authentic_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(authentic_dir, f), 0))
        
        if os.path.exists(forged_dir):
            for f in os.listdir(forged_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(forged_dir, f), 1))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
