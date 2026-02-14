from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import SkinAnalysisResult, RecommendationRequest, MatchResult
from app.services.ai_service import analyze_face_image, get_shade_recommendation
from fastapi import APIRouter, UploadFile, File, Depends
from sqlmodel import Session
from app.database import get_session  
from app.models import UserProfile

router = APIRouter()

@router.post("/analyze")
async def analyze_skin(
    file: UploadFile = File(...), 
    db: Session = Depends(get_session) 
):
    contents = await file.read()
    ai_result = await analyze_face_image(contents)
    
    new_profile = UserProfile(
        user_id="test_user_123", 
        skin_tone=ai_result["skin_tone"],
        undertone=ai_result["undertone"],
        summary=ai_result["summary"]
    )
    db.add(new_profile)
    db.commit()
    db.refresh(new_profile)
    
    return new_profile

@router.post("/recommend", response_model=MatchResult)
async def recommend_product(request: RecommendationRequest):
    """
    Send Profile + List of Products -> Get Best Match
    """
    result = await get_shade_recommendation(request.user_profile, request.products)
    return result