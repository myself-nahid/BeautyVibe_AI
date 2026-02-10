from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas import SkinAnalysisResult, RecommendationRequest, MatchResult
from app.services.ai_service import analyze_face_image, get_shade_recommendation

router = APIRouter()

@router.post("/analyze", response_model=SkinAnalysisResult)
async def analyze_skin(file: UploadFile = File(...)):
    """
    Upload a selfie -> Get Skin Profile
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    contents = await file.read()
    result = await analyze_face_image(contents)
    return result

@router.post("/recommend", response_model=MatchResult)
async def recommend_product(request: RecommendationRequest):
    """
    Send Profile + List of Products -> Get Best Match
    """
    result = await get_shade_recommendation(request.user_profile, request.products)
    return result