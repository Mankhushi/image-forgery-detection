"""Main ML Pipeline for Image Forgery Detection"""
import torch
import numpy as np
from PIL import Image
from typing import Dict, Union
from .preprocessing import ImageProcessor, ELAProcessor
from .models import EnsembleDetector
from .postprocessing import HeatmapGenerator, ConfidenceScorer

class ForgeryDetectionPipeline:
    def __init__(self, model_paths: Dict[str, str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = ImageProcessor()
        self.ela_processor = ELAProcessor()
        self.heatmap_generator = HeatmapGenerator()
        self.confidence_scorer = ConfidenceScorer()
        self.model = EnsembleDetector()
        
        if model_paths:
            self.model.load_all_weights(model_paths)
        
        self.model.eval()
    
    def analyze(self, image: Union[str, bytes, np.ndarray, Image.Image]) -> Dict:
        """Complete forgery analysis pipeline"""
        # Load and preprocess
        pil_image = self.image_processor.load_image(image)
        np_image = np.array(pil_image)
        tensor = self.image_processor.preprocess(pil_image)
        
        # Model prediction
        prediction = self.model.predict(tensor)
        
        # ELA analysis
        ela_analysis = self.ela_processor.analyze(np_image)
        
        # Generate report
        report = self.confidence_scorer.generate_report_data(prediction, ela_analysis)
        
        return {
            'verdict': report['verdict'],
            'confidence': report['confidence'],
            'score': report['combined_score'],
            'details': report['details'],
            'suspicious_regions': report['suspicious_regions'],
            'heatmap': ela_analysis['heatmap'],
            'ela_image': ela_analysis['ela']
        }
    
    def quick_check(self, image: Union[str, bytes, np.ndarray]) -> Dict:
        """Quick forgery check without full analysis"""
        tensor = self.image_processor.preprocess(image)
        return self.model.predict(tensor)
