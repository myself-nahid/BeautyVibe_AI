"""
app/models.py
─────────────
SQLModel table definitions.

Each class that carries  table=True  maps to a database table.
Fields without defaults are required on INSERT.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

class UserProfile(SQLModel, table=True):
    """
    Stores the AI-analysed beauty profile for a user.
    One user may have multiple profiles (re-analysed over time).
    """

    __tablename__ = "user_profiles"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Identifier supplied by the calling client (auth token subject, device id, etc.)
    user_id: str = Field(index=True, max_length=128)

    # Core AI analysis results
    skin_tone: str = Field(max_length=32)    # Fair | Light | Medium | Tan | Deep
    undertone: str = Field(max_length=32)    # Cool | Neutral | Warm
    face_shape: str = Field(max_length=32)   # Oval | Round | Square | Heart | Diamond | Oblong
    eye_color: str = Field(default="Unknown", max_length=64)

    # AI confidence and human-readable summary
    confidence_score: int = Field(default=0, ge=0, le=100)
    summary: str = Field(default="")

    # Audit timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

class Product(SQLModel, table=True):
    """
    Beauty product catalogue entry managed by sellers / admin.
    Covers foundations, lipsticks, blushers, serums, etc.
    """

    __tablename__ = "products"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Seller / brand reference
    seller_id: str = Field(index=True, max_length=128)

    # Product identity
    name: str = Field(max_length=256)
    sku: Optional[str] = Field(default=None, max_length=128, index=True)
    category: str = Field(max_length=64)      # Foundation | Lipstick | Blush …
    shade_name: Optional[str] = Field(default=None, max_length=128)
    hex_code: Optional[str] = Field(default=None, max_length=7)  # "#FFDAB9"

    # Pricing
    price: float = Field(ge=0)
    currency: str = Field(default="USD", max_length=8)

    # Content & availability
    description: Optional[str] = Field(default=None)
    image_url: Optional[str] = Field(default=None, max_length=512)
    is_active: bool = Field(default=True)

    # Audit timestamps
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

class RecommendationLog(SQLModel, table=True):
    """
    Persists every shade-recommendation event for analytics and
    commission tracking (as required by the GlowFlow spec).
    """

    __tablename__ = "recommendation_logs"

    id: Optional[int] = Field(default=None, primary_key=True)

    user_id: str = Field(index=True, max_length=128)
    best_match_product_id: str = Field(max_length=128)
    match_score: int = Field(ge=0, le=100)
    reasoning: str = Field(default="")

    # Snapshot of the profile used (denormalised for historical accuracy)
    skin_tone: str = Field(max_length=32)
    undertone: str = Field(max_length=32)
    eye_color: str = Field(default="Unknown", max_length=64)

    created_at: datetime = Field(default_factory=_utcnow)