from fastapi import APIRouter, Depends, File, UploadFile, status, Body
from sqlmodel import Session
from app.database import get_session
from app.core.security import verify_api_key  # <--- Import this
from app.services.ai_service import analyze_face_image, get_shade_recommendation
from app.schemas import UserProfileResponse, MatchResult, RecommendationRequest, SkinAnalysisResult

# Lock down the entire router
router = APIRouter(tags=["Internal AI Service"], dependencies=[Depends(verify_api_key)])

@router.post(
    "/analyze",
    response_model=SkinAnalysisResult, # Return pure analysis data
    summary="Server-to-Server Face Analysis",
)
async def analyze_skin(
    image: UploadFile = File(..., description="User's face image for analysis"),
):
    """
    Receives an image file from the Main Backend.
    Returns the raw analysis JSON (Skin Tone, Undertone, etc.).
    """
    image_bytes = await image.read()
    
    # AI Service analyzes the image
    ai_result = await analyze_face_image(image_bytes)
    
    # We return the dictionary directly. 
    # The Main Backend will decide whether to save this to their DB.
    return ai_result


@router.post(
    "/recommend",
    response_model=MatchResult,
    summary="Server-to-Server Recommendation",
)
async def recommend_product(
    request: RecommendationRequest,
):
    """
    Receives a User Profile + Product List from Main Backend.
    Returns the best match.
    """
    result = await get_shade_recommendation(request.user_profile, request.products)
    return result