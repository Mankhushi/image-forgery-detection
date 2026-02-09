"""Ensemble Model combining all detectors"""
import torch
import torch.nn as nn
from typing import Dict, List
from .copy_move_detector import CopyMoveDetector
from .splicing_detector import SplicingDetector
from .deepfake_detector import DeepfakeDetector

class EnsembleDetector(nn.Module):
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.copy_move = CopyMoveDetector()
        self.splicing = SplicingDetector()
        self.deepfake = DeepfakeDetector()
        
        self.weights = weights or {'copy_move': 0.35, 'splicing': 0.35, 'deepfake': 0.30}
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> Dict:
        self.eval()
        with torch.no_grad():
            cm_out = torch.softmax(self.copy_move(x), dim=1)
            sp_out = torch.softmax(self.splicing(x), dim=1)
            df_out = torch.softmax(self.deepfake(x), dim=1)
        
        # Weighted average
        ensemble_prob = (
            self.weights['copy_move'] * cm_out +
            self.weights['splicing'] * sp_out +
            self.weights['deepfake'] * df_out
        )
        
        return {
            'ensemble': ensemble_prob,
            'copy_move': cm_out,
            'splicing': sp_out,
            'deepfake': df_out
        }
    
    def predict(self, image: torch.Tensor) -> Dict:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        outputs = self.forward(image)
        ensemble_prob = outputs['ensemble'][0]
        
        is_forged = ensemble_prob[1] > ensemble_prob[0]
        confidence = ensemble_prob[1].item() if is_forged else ensemble_prob[0].item()
        
        return {
            'prediction': 'forged' if is_forged else 'authentic',
            'confidence': confidence,
            'forgery_probability': ensemble_prob[1].item(),
            'model_scores': {
                'copy_move': outputs['copy_move'][0, 1].item(),
                'splicing': outputs['splicing'][0, 1].item(),
                'deepfake': outputs['deepfake'][0, 1].item()
            }
        }
    
    def load_all_weights(self, paths: Dict[str, str]):
        if 'copy_move' in paths:
            self.copy_move.load_weights(paths['copy_move'])
        if 'splicing' in paths:
            self.splicing.load_weights(paths['splicing'])
        if 'deepfake' in paths:
            self.deepfake.load_weights(paths['deepfake'])
