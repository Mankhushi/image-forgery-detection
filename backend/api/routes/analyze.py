"""Image Analysis Routes"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
import os
import uuid
import cv2
from PIL import Image
import numpy as np

from ...database import get_db
from ...models.user import User
from ...models.analysis import Analysis
from ...config import settings
from .auth import get_current_user, get_current_user_optional

# Import ML pipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from ml.pipeline import ForgeryDetectionPipeline

router = APIRouter(prefix="/analyze", tags=["Analysis"])

# Initialize pipeline (lazy loading)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = ForgeryDetectionPipeline()
    return _pipeline

@router.post("")
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Save uploaded file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, 'wb') as f:
        f.write(contents)
    
    # Analyze image
    pipeline = get_pipeline()
    result = pipeline.analyze(contents)
    
    # Save heatmap
    heatmap_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}_heatmap.png")
    cv2.imwrite(heatmap_path, cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR))
    
    # Save to database
    analysis = Analysis(
        user_id=current_user.id if current_user else None,
        filename=file.filename,
        file_path=file_path,
        verdict=result['verdict'],
        confidence=result['confidence'],
        score=result['score'],
        details=result['details'],
        heatmap_path=heatmap_path
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    return {
        "id": analysis.id,
        "verdict": result['verdict'],
        "confidence": result['confidence'],
        "score": result['score'],
        "details": result['details'],
        "suspicious_regions": result['suspicious_regions']
    }

@router.get("/{analysis_id}/heatmap")
async def get_heatmap(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if not os.path.exists(analysis.heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    return FileResponse(analysis.heatmap_path, media_type="image/png")

@router.post("/quick")
async def quick_analyze(file: UploadFile = File(...)):
    """Quick analysis without saving to database"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    pipeline = get_pipeline()
    result = pipeline.quick_check(contents)
    
    return result
