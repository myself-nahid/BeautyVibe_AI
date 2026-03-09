from fastapi import APIRouter, Depends, File, UploadFile

from app.core.security import verify_api_key
from app.services.ai_service import analyze_face_image, get_shade_recommendations
from app.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    SkinAnalysisResult,
)

# Every route in this router requires a valid X-API-KEY header
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
    Returns the raw skin analysis JSON (Skin Tone, Undertone, Face Shape, etc.).
    The Main Backend decides whether to persist the result.
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
    Accepts the EXACT explore-page payload from the Main Backend and
    returns a ranked list of the best-matching product shades.

    The `categories` array is joined with the `products` array here so
    the AI receives a fully enriched product list (with category names).

    Request body shape
    ──────────────────
    {
      "user_id":      "test_user_123",
      "top_n":        3,
      "user_profile": {
        "skin_tone":        "Deep",
        "undertone":        "Warm",
        "face_shape":       "Oval",
        "eye_color":        "Brown",
        "confidence_score": 98,
        "summary":          "Deep skin with warm undertones."
      },
      "categories": [
        { "id": 1, "name": "Lipstick", "slug": "lipstick", ... },
        { "id": 2, "name": "Cream",    "slug": "cream",    ... }
      ],
      "products": [
        {
          "id":                  16,
          "name":                "Velvet Matte Lipstick",
          "brand":               "VELYVA",
          "shade":               "Crystal Pink",
          "price":               "399.00",
          "discount_percentage": 0,
          "rating":              "0.0",
          "image":               "http://..."
        },
        ...
      ]
    }

    Response shape
    ──────────────
    {
      "recommendations": [
        {
          "best_match_id":   "3",
          "match_score":     94,
          "reasoning":       "Peachy Blush ...",
          "matched_product": { ... }
        }
      ],
      "total": 3
    }
    """
    # Resolve category names onto each product before sending to AI
    enriched_products = request.products_with_category()

    return await get_shade_recommendations(
        profile=request.user_profile,
        products=enriched_products,
        top_n=request.top_n,
    )