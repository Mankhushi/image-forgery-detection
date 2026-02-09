"""History Routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from datetime import datetime

from ...database import get_db
from ...models.user import User
from ...models.analysis import Analysis
from .auth import get_current_user

router = APIRouter(prefix="/history", tags=["History"])

class AnalysisResponse(BaseModel):
    id: int
    filename: str
    verdict: str
    confidence: str
    score: float
    created_at: datetime
    
    class Config:
        from_attributes = True

@router.get("", response_model=List[AnalysisResponse])
def get_history(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    analyses = db.query(Analysis).filter(
        Analysis.user_id == current_user.id
    ).order_by(Analysis.created_at.desc()).offset(skip).limit(limit).all()
    
    return analyses

@router.get("/{analysis_id}")
def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "id": analysis.id,
        "filename": analysis.filename,
        "verdict": analysis.verdict,
        "confidence": analysis.confidence,
        "score": analysis.score,
        "details": analysis.details,
        "created_at": analysis.created_at
    }

@router.delete("/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == current_user.id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    db.delete(analysis)
    db.commit()
    
    return {"message": "Analysis deleted successfully"}

@router.get("/stats/summary")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    total = db.query(Analysis).filter(Analysis.user_id == current_user.id).count()
    forged = db.query(Analysis).filter(
        Analysis.user_id == current_user.id,
        Analysis.verdict == "FORGED"
    ).count()
    authentic = total - forged
    
    return {
        "total_analyses": total,
        "forged_count": forged,
        "authentic_count": authentic,
        "forgery_rate": forged / total if total > 0 else 0
    }
