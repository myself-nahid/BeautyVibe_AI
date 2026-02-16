"""
app/schemas.py
──────────────
Pydantic v2 request / response schemas (DTOs).

Kept separate from SQLModel table models so that the API contract
can evolve independently of the persistence layer.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

SKIN_TONES = {"Fair", "Light", "Medium", "Tan", "Deep"}
UNDERTONES = {"Cool", "Neutral", "Warm"}
FACE_SHAPES = {"Oval", "Round", "Square", "Heart", "Diamond", "Oblong"}


class SkinAnalysisResult(BaseModel):
    """Response schema for the /analyze endpoint."""

    skin_tone: str = Field(examples=["Medium"])
    undertone: str = Field(examples=["Warm"])
    face_shape: str = Field(examples=["Oval"])
    eye_color: str = Field(default="Unknown", examples=["Brown"])
    confidence_score: int = Field(default=0, ge=0, le=100, examples=[85])
    summary: str = Field(default="", examples=["Medium skin with warm undertones and oval face shape."])

    @field_validator("skin_tone")
    @classmethod
    def validate_skin_tone(cls, v: str) -> str:
        val = v.strip().title()
        if val not in SKIN_TONES:
            # Gracefully fall back rather than hard-reject
            return "Medium"
        return val

    @field_validator("undertone")
    @classmethod
    def validate_undertone(cls, v: str) -> str:
        val = v.strip().title()
        if val not in UNDERTONES:
            return "Neutral"
        return val

    @field_validator("face_shape")
    @classmethod
    def validate_face_shape(cls, v: str) -> str:
        val = v.strip().title()
        if val not in FACE_SHAPES:
            return "Oval"
        return val

class UserProfileResponse(BaseModel):
    """Public representation of a saved UserProfile row."""

    id: int
    user_id: str
    skin_tone: str
    undertone: str
    face_shape: str
    eye_color: str
    confidence_score: int
    summary: str
    created_at: datetime

    model_config = {"from_attributes": True}

class ProductShade(BaseModel):
    """A single product/shade entry supplied by the caller for matching."""

    id: str = Field(examples=["prod_001"])
    name: str = Field(examples=["Peachy Blush"])
    hex_code: str = Field(examples=["#FFDAB9"])
    category: str = Field(examples=["Lipstick"])
    price: Optional[float] = Field(default=None, examples=[24.99])
    description: Optional[str] = Field(default=None)

    @field_validator("hex_code")
    @classmethod
    def validate_hex(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("#") or len(v) not in (4, 7):
            raise ValueError(f"'{v}' is not a valid hex colour code (e.g. #FFDAB9).")
        return v.upper()

class RecommendationRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier — used to persist the recommendation log.",
        examples=["user_abc123"],
    )
    user_profile: SkinAnalysisResult
    products: List[ProductShade] = Field(
        min_length=1,
        description="At least one product must be supplied.",
    )


class MatchResult(BaseModel):
    """Response schema for the /recommend endpoint."""

    best_match_id: str
    match_score: int = Field(ge=0, le=100)
    reasoning: str
    matched_product: Optional[ProductShade] = Field(
        default=None,
        description="Full product details for the matched item.",
    )

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str