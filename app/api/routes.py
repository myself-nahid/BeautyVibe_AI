"""
app/api/v1/routes.py
────────────────────
GlowFlow API v1 endpoints.

Endpoints
─────────
POST /analyze
    • Accepts a face image (JPEG / PNG / WebP, ≤ 5 MB).
    • Calls the AI vision service to classify skin tone, undertone,
      face shape, and eye colour.
    • Persists the result as a UserProfile row.
    • Returns the saved profile.

POST /recommend
    • Accepts a beauty profile + list of product shades.
    • Calls the AI to select the best-matching product.
    • Optionally persists a RecommendationLog row (if user_id is supplied).
    • Returns the matched product with a score and colour-theory reasoning.

GET /profiles/{user_id}
    • Returns all saved beauty profiles for the given user.

GET /health
    • Lightweight liveness probe (no DB query).
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, File, Query, UploadFile, status
from sqlmodel import Session, select

from app.core.config import get_settings
from app.database import get_session
from app.models import RecommendationLog, UserProfile
from app.schemas import (
    MatchResult,
    RecommendationRequest,
    UserProfileResponse,
)
from app.services.ai_service import analyze_face_image, get_shade_recommendation

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["GlowFlow AI"])

@router.post(
    "/analyze",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse a face image",
    description=(
        "Upload a face image (JPEG, PNG, or WebP ≤ 5 MB). "
        "The AI classifies skin tone, undertone, face shape, and eye colour, "
        "then stores and returns the beauty profile."
    ),
)
async def analyze_skin(
    file: UploadFile = File(..., description="Face image — JPEG, PNG, or WebP"),
    user_id: str = Query(
        default="guest",
        description="Caller-supplied user identifier (e.g. auth token subject).",
        max_length=128,
    ),
    db: Session = Depends(get_session),
) -> UserProfileResponse:
    """Analyse a face image and persist + return the beauty profile."""

    logger.info("analyze_skin: user_id=%s file=%s", user_id, file.filename)

    image_bytes = await file.read()

    # Delegate validation + AI call to the service layer
    ai_result = await analyze_face_image(image_bytes)

    profile = UserProfile(
        user_id=user_id,
        skin_tone=ai_result["skin_tone"],
        undertone=ai_result["undertone"],
        face_shape=ai_result["face_shape"],
        eye_color=ai_result.get("eye_color", "Unknown"),
        confidence_score=ai_result.get("confidence_score", 0),
        summary=ai_result.get("summary", ""),
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    logger.info("Profile saved: id=%s user_id=%s", profile.id, profile.user_id)
    return UserProfileResponse.model_validate(profile)

@router.post(
    "/recommend",
    response_model=MatchResult,
    status_code=status.HTTP_200_OK,
    summary="Get a shade recommendation",
    description=(
        "Send a beauty profile and a list of product shades. "
        "The AI returns the single best-matching product with a score "
        "and colour-theory reasoning."
    ),
)
async def recommend_product(
    request: RecommendationRequest,
    db: Session = Depends(get_session),
) -> MatchResult:
    """Match a beauty profile against a product list and return the best shade."""

    logger.info(
        "recommend_product: user_id=%s products=%d",
        request.user_id,
        len(request.products),
    )

    result = await get_shade_recommendation(request.user_profile, request.products)

    # Persist recommendation for analytics / commission tracking
    if request.user_id:
        log_entry = RecommendationLog(
            user_id=request.user_id,
            best_match_product_id=result.best_match_id,
            match_score=result.match_score,
            reasoning=result.reasoning,
            skin_tone=request.user_profile.skin_tone,
            undertone=request.user_profile.undertone,
            eye_color=request.user_profile.eye_color,
        )
        db.add(log_entry)
        db.commit()
        logger.info(
            "RecommendationLog saved: product_id=%s score=%s",
            result.best_match_id,
            result.match_score,
        )

    return result

@router.get(
    "/profiles/{user_id}",
    response_model=List[UserProfileResponse],
    status_code=status.HTTP_200_OK,
    summary="Get beauty profiles for a user",
    description="Returns all saved beauty profiles for the given user_id, newest first.",
)
async def get_user_profiles(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of profiles to return."),
    db: Session = Depends(get_session),
) -> List[UserProfileResponse]:
    """Return paginated beauty profiles for the specified user."""

    statement = (
        select(UserProfile)
        .where(UserProfile.user_id == user_id)
        .order_by(UserProfile.created_at.desc())  # type: ignore[arg-type]
        .limit(limit)
    )
    profiles = db.exec(statement).all()
    return [UserProfileResponse.model_validate(p) for p in profiles]