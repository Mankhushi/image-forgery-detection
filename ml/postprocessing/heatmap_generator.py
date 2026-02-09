"""Heatmap Generator for visualizing forgery regions"""
import cv2
import numpy as np
import torch
from typing import Tuple, Optional

class HeatmapGenerator:
    def __init__(self, colormap: int = cv2.COLORMAP_JET):
        self.colormap = colormap
    
    def generate_cam(self, model, image: torch.Tensor, target_layer) -> np.ndarray:
        """Generate Class Activation Map"""
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])
        
        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)
        
        model.eval()
        output = model(image)
        pred_class = output.argmax(dim=1)
        
        model.zero_grad()
        output[0, pred_class].backward()
        
        handle_f.remove()
        handle_b.remove()
        
        # Generate CAM
        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    
    def apply_heatmap(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay heatmap on image"""
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, self.colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    def get_bounding_boxes(self, cam: np.ndarray, threshold: float = 0.5, min_area: int = 100) -> list:
        """Extract bounding boxes from CAM"""
        h, w = cam.shape
        cam_uint8 = (cam * 255).astype(np.uint8)
        _, binary = cv2.threshold(cam_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                boxes.append({'x': int(x), 'y': int(y), 'width': int(bw), 'height': int(bh)})
        return boxes
