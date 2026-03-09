"""
app/schemas.py
──────────────
Pydantic v2 request / response schemas (DTOs).
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
# Category
# ─────────────────────────────────────────────

class Category(BaseModel):
    """Matches the exact shape in the backend's categories array."""
    id:         int
    name:       str
    slug:       str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ─────────────────────────────────────────────
# Product  (input — from backend payload)
# ─────────────────────────────────────────────

class ProductShade(BaseModel):
    """
    Matches the EXACT product object shape from the backend payload.

    Coercions:
      • id    : int  → str
      • price : str  → float
    """
    id:                  str             = Field(examples=["16"])
    name:                str             = Field(examples=["Velvet Matte Lipstick"])
    slug:                Optional[str]   = Field(default=None)
    brand:               Optional[str]   = Field(default=None, examples=["VELYVA"])
    shade:               Optional[str]   = Field(default=None, examples=["Crystal Pink"])
    price:               Optional[float] = Field(default=None, examples=[399.00])
    discount_percentage: Optional[int]   = Field(default=0,    examples=[20])
    rating:              Optional[float]   = Field(default=None, examples=["4.5"])
    image:               Optional[str]   = Field(default=None)
    description:         Optional[str]   = Field(default=None)
    hex_code:            Optional[str]   = Field(default=None)
    # Resolved internally — never in raw product objects
    category:            Optional[str]   = Field(default=None)

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id_to_str(cls, v) -> str:
        return str(v)

    @field_validator("price", mode="before")
    @classmethod
    def coerce_price_to_float(cls, v) -> Optional[float]:
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
            raise ValueError(f"'{v}' is not a valid hex colour code.")
        return v.upper()


# ─────────────────────────────────────────────
# Matched product  (output — clean response shape)
# ─────────────────────────────────────────────

class MatchedProduct(BaseModel):
    """
    Product fields returned inside each recommendation.
    Mirrors the exact fields from the backend's products array.
    No internal fields (category, hex_code, description).
    """
    id:                  str
    name:                str
    slug:                Optional[str]   = None
    brand:               Optional[str]   = None
    shade:               Optional[str]   = None
    price:               Optional[float] = None
    discount_percentage: Optional[int]   = None
    rating:              Optional[float] = None
    image:               Optional[str]   = None
    category:            Optional[str]   = None
    hex_code:            Optional[str]   = None

    @classmethod
    def from_product_shade(cls, p: ProductShade) -> "MatchedProduct":
        return cls(
            id=p.id,
            name=p.name,
            slug=p.slug,
            brand=p.brand,
            shade=p.shade,
            price=p.price,
            discount_percentage=p.discount_percentage,
            rating=p.rating,
            image=p.image,
            category=p.category,
            hex_code=p.hex_code,
        )


# ─────────────────────────────────────────────
# Recommendation request
# ─────────────────────────────────────────────

class ExploreData(BaseModel):
    """The inner `data` object of the Main Backend's explore response."""
    user_id:        Optional[str]      = None
    user_profile:   SkinAnalysisResult
    categories:     List[Category]     = Field(default_factory=list)
    products:       List[ProductShade] = Field(min_length=1)
    total_products: Optional[int]      = None
    page:           Optional[int]      = None
    total_pages:    Optional[int]      = None
    next:           Optional[str]      = None
    previous:       Optional[str]      = None


class RecommendationRequest(BaseModel):
    """
    Accepts the full backend payload in either shape:

    Shape 1 — Full envelope:
      { "success": true, "status": 200, "message": "...", "top_n": 3, "data": { ... } }

    Shape 2 — Flat (inner data directly):
      { "user_profile": {...}, "categories": [...], "products": [...], "top_n": 3 }
    """
    # Outer envelope
    success: Optional[bool]        = None
    status:  Optional[int]         = None
    message: Optional[str]         = None
    data:    Optional[ExploreData] = None

    # Flat fields
    user_id:        Optional[str]               = None
    user_profile:   Optional[SkinAnalysisResult]= None
    categories:     List[Category]              = Field(default_factory=list)
    products:       Optional[List[ProductShade]]= None
    total_products: Optional[int]               = None
    page:           Optional[int]               = None
    total_pages:    Optional[int]               = None
    next:           Optional[str]               = None
    previous:       Optional[str]               = None

    top_n: int = Field(default=3, ge=1, le=20)

    def get_explore_data(self) -> ExploreData:
        if self.data is not None:
            return self.data
        if self.user_profile is not None and self.products is not None:
            return ExploreData(
                user_id=self.user_id,
                user_profile=self.user_profile,
                categories=self.categories,
                products=self.products,
                total_products=self.total_products,
                page=self.page,
                total_pages=self.total_pages,
                next=self.next,
                previous=self.previous,
            )
        raise ValueError(
            "Request must include either a 'data' object or flat "
            "'user_profile' + 'products' fields."
        )

    def products_with_category(self) -> List[ProductShade]:
        """
        Resolve category name onto each product from the categories array.

        Matching order (first match wins):
          1. Category slug is a substring of the product slug.
          2. Category name is a substring of the product name.
          3. Falls back to the first category in the list.
        """
        explore    = self.get_explore_data()
        products   = explore.products
        categories = explore.categories

        if not categories:
            return products

        fallback = categories[0].name
        enriched: List[ProductShade] = []

        for product in products:
            if product.category:
                enriched.append(product)
                continue

            resolved = fallback

            if product.slug:
                slug_lower = product.slug.lower()
                for cat in categories:
                    if cat.slug.lower() in slug_lower:
                        resolved = cat.name
                        break
                else:
                    name_lower = product.name.lower()
                    for cat in categories:
                        if cat.name.lower() in name_lower:
                            resolved = cat.name
                            break
            else:
                name_lower = product.name.lower()
                for cat in categories:
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
    matched_product: Optional[MatchedProduct] = None


class RecommendationResponse(BaseModel):
    """
    Response schema for the /recommend endpoint.

    categories  — the original category list passed through unchanged.
    recommendations — ranked product matches (no category inside each product).
    total       — number of recommendations returned.
    """
    categories:      List[Category]
    recommendations: List[MatchResult]
    total:           int


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    version:     str
    environment: str