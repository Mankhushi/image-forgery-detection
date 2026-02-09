"""Train forgery detection model"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from torch.utils.data import DataLoader, random_split
from ml.models import CopyMoveDetector, SplicingDetector
from ml.training import ModelTrainer, ForgeryDataset

def train(data_dir: str, model_type: str = "copy_move", epochs: int = 50):
    # Load dataset
    dataset = ForgeryDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Select model
    if model_type == "copy_move":
        model = CopyMoveDetector()
    elif model_type == "splicing":
        model = SplicingDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    trainer = ModelTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=epochs, save_path=f"models/{model_type}")
    
    print(f"Training complete! Best model saved to models/{model_type}/best_model.pth")
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--model", default="copy_move", choices=["copy_move", "splicing", "deepfake"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train(args.data, args.model, args.epochs)
