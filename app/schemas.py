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

SKIN_TONES  = {"Fair", "Light", "Medium", "Tan", "Deep"}
UNDERTONES  = {"Cool", "Neutral", "Warm"}
FACE_SHAPES = {"Oval", "Round", "Square", "Heart", "Diamond", "Oblong"}


# ─────────────────────────────────────────────
# Skin / face analysis
# ─────────────────────────────────────────────

class SkinAnalysisResult(BaseModel):
    """Response schema for the /analyze endpoint."""

    skin_tone:        str = Field(examples=["Medium"])
    undertone:        str = Field(examples=["Warm"])
    face_shape:       str = Field(examples=["Oval"])
    eye_color:        str = Field(default="Unknown", examples=["Brown"])
    confidence_score: int = Field(default=0, ge=0, le=100, examples=[85])
    summary:          str = Field(default="", examples=["Medium skin with warm undertones."])

    @field_validator("skin_tone")
    @classmethod
    def validate_skin_tone(cls, v: str) -> str:
        val = v.strip().title()
        return val if val in SKIN_TONES else "Medium"

    @field_validator("undertone")
    @classmethod
    def validate_undertone(cls, v: str) -> str:
        val = v.strip().title()
        return val if val in UNDERTONES else "Neutral"

    @field_validator("face_shape")
    @classmethod
    def validate_face_shape(cls, v: str) -> str:
        val = v.strip().title()
        return val if val in FACE_SHAPES else "Oval"


class UserProfileResponse(BaseModel):
    """Public representation of a saved UserProfile row."""

    id:               int
    user_id:          str
    skin_tone:        str
    undertone:        str
    face_shape:       str
    eye_color:        str
    confidence_score: int
    summary:          str
    created_at:       datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────
# Category  (matches backend categories array)
# ─────────────────────────────────────────────

class Category(BaseModel):
    """
    One entry from the backend's `categories` array.
    Matches the exact shape sent by the Main Backend.
    """
    id:         int
    name:       str
    slug:       str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ─────────────────────────────────────────────
# Product  (matches backend products array)
# ─────────────────────────────────────────────

class ProductShade(BaseModel):
    """
    One product entry from the Main Backend's products array.

    • id               – sent as int by backend, coerced to str internally
    • price            – sent as string ("399.00"), coerced to float
    • category         – NOT in the product object; resolved from the
                         categories array by RecommendationRequest and
                         injected here before AI processing.
    """
    id:                  str            = Field(examples=["16"])
    name:                str            = Field(examples=["Velvet Matte Lipstick"])
    slug:                Optional[str]  = Field(default=None)
    brand:               Optional[str]  = Field(default=None, examples=["VELYVA"])
    shade:               Optional[str]  = Field(default=None, examples=["Crystal Pink"])
    price:               Optional[float]= Field(default=None, examples=[399.00])
    discount_percentage: Optional[int]  = Field(default=0,    examples=[20])
    rating:              Optional[str]  = Field(default=None, examples=["4.5"])
    image:               Optional[str]  = Field(default=None)
    description:         Optional[str]  = Field(default=None)
    hex_code:            Optional[str]  = Field(default=None, examples=["#FFDAB9"])
    # Injected from the categories array — not present in raw product objects
    category:            Optional[str]  = Field(default=None, examples=["Lipstick"])

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v) -> str:
        """Backend sends id as integer — normalise to string."""
        return str(v)

    @field_validator("price", mode="before")
    @classmethod
    def coerce_price_to_float(cls, v) -> Optional[float]:
        """Backend sends price as string e.g. '399.00' — normalise to float."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @field_validator("hex_code")
    @classmethod
    def validate_hex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v.startswith("#") or len(v) not in (4, 7):
            raise ValueError(f"'{v}' is not a valid hex colour code (e.g. #FFDAB9).")
        return v.upper()


# ─────────────────────────────────────────────
# Recommendation request  (full explore payload)
# ─────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """
    Accepts the EXACT explore-page payload shape sent by the Main Backend:

        {
          "user_id": "...",
          "user_profile": { ... },
          "categories": [ { "id": 1, "name": "Lipstick", "slug": "lipstick", ... } ],
          "products": [ { "id": 16, "name": "...", "shade": "...", ... } ],
          "top_n": 3          ← optional, default 3
        }

    products_with_category() resolves the missing category field on each
    product by matching category names against the product name.
    """

    user_id:      Optional[str]      = Field(default=None, examples=["user_abc123"])
    user_profile: SkinAnalysisResult
    categories:   List[Category]     = Field(default_factory=list)
    products:     List[ProductShade] = Field(min_length=1)
    top_n:        int                = Field(default=3, ge=1, le=20)

    # Pagination fields — accepted so validation never fails, not used by AI
    total_products: Optional[int] = None
    page:           Optional[int] = None
    total_pages:    Optional[int] = None
    next:           Optional[str] = None
    previous:       Optional[str] = None

    def products_with_category(self) -> List[ProductShade]:
        """
        Return the product list with `category` resolved from the
        categories array.

        Matching strategy (first match wins):
          1. Category name is a substring of product name (case-insensitive).
          2. Falls back to the first category in the list.
          3. Falls back to None if categories list is empty.
        """
        if not self.categories:
            return self.products

        fallback = self.categories[0].name
        enriched: List[ProductShade] = []

        for product in self.products:
            if product.category:
                enriched.append(product)
                continue

            resolved = fallback
            name_lower = product.name.lower()
            for cat in self.categories:
                if cat.name.lower() in name_lower:
                    resolved = cat.name
                    break

            enriched.append(product.model_copy(update={"category": resolved}))

        return enriched


# ─────────────────────────────────────────────
# Recommendation response
# ─────────────────────────────────────────────

class MatchResult(BaseModel):
    """A single ranked recommendation entry."""

    best_match_id:   str
    match_score:     int = Field(ge=0, le=100)
    reasoning:       str
    matched_product: Optional[ProductShade] = Field(
        default=None,
        description="Full product details for the matched item.",
    )


class RecommendationResponse(BaseModel):
    """Response schema for the /recommend endpoint."""

    recommendations: List[MatchResult]
    total:           int = Field(description="Number of recommendations returned.")


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    version:     str
    environment: str