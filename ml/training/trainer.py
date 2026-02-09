"""Model Trainer"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional
import os
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        for images, labels in tqdm(dataloader, desc='Training'):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return {'loss': total_loss / len(dataloader), 'accuracy': correct / total}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, save_path: str = 'models'):
        os.makedirs(save_path, exist_ok=True)
        best_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            self.scheduler.step(val_metrics['loss'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(), os.path.join(save_path, 'best_model.pth'))
                print(f"Saved best model with accuracy: {best_acc:.4f}")
        
        return self.history
