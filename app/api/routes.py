from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.security import verify_api_key
from app.services.ai_service import analyze_face_image, get_shade_recommendations
from app.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    SkinAnalysisResult,
)

router = APIRouter(
    tags=["Internal AI Service"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/analyze",
    response_model=SkinAnalysisResult,
    summary="Server-to-Server Face Analysis",
)
async def analyze_skin(
    image: UploadFile = File(..., description="User's face image for analysis"),
):
    """
    Receives an image file from the Main Backend.
    Returns the raw skin analysis JSON.
    """
    image_bytes = await image.read()
    return await analyze_face_image(image_bytes)


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Server-to-Server Shade Recommendations",
)
async def recommend_products(
    request: RecommendationRequest,
):
    """
    Accepts the Main Backend's explore payload and returns ranked
    recommendations with categories as a separate top-level list.
    """
    try:
        explore = request.get_explore_data()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    enriched_products = request.products_with_category()

    return await get_shade_recommendations(
        profile=explore.user_profile,
        products=enriched_products,
        categories=explore.categories,   # passed separately, returned separately
        top_n=request.top_n,
    )