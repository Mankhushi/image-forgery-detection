"""Confidence Scoring Module"""
import numpy as np
from typing import Dict

class ConfidenceScorer:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            'high': 0.85,
            'medium': 0.65,
            'low': 0.50
        }
    
    def get_confidence_level(self, score: float) -> str:
        if score >= self.thresholds['high']:
            return 'high'
        elif score >= self.thresholds['medium']:
            return 'medium'
        elif score >= self.thresholds['low']:
            return 'low'
        return 'very_low'
    
    def calculate_combined_score(self, model_scores: Dict[str, float], ela_score: float) -> Dict:
        model_avg = np.mean(list(model_scores.values()))
        combined = 0.7 * model_avg + 0.3 * ela_score
        
        return {
            'combined_score': float(combined),
            'model_average': float(model_avg),
            'ela_score': float(ela_score),
            'confidence_level': self.get_confidence_level(combined),
            'is_likely_forged': combined > 0.5
        }
    
    def generate_report_data(self, prediction: Dict, ela_analysis: Dict) -> Dict:
        model_scores = prediction.get('model_scores', {})
        ela_score = ela_analysis.get('statistics', {}).get('suspicious_score', 0)
        
        scoring = self.calculate_combined_score(model_scores, ela_score)
        
        return {
            'verdict': 'FORGED' if scoring['is_likely_forged'] else 'AUTHENTIC',
            'confidence': scoring['confidence_level'],
            'combined_score': scoring['combined_score'],
            'details': {
                'copy_move_score': model_scores.get('copy_move', 0),
                'splicing_score': model_scores.get('splicing', 0),
                'deepfake_score': model_scores.get('deepfake', 0),
                'ela_score': ela_score
            },
            'suspicious_regions': ela_analysis.get('suspicious_regions', [])
        }
