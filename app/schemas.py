from pydantic import BaseModel
from typing import List, Optional

# AI Analysis Response
class SkinAnalysisResult(BaseModel):
    skin_tone: str       # e.g., "Medium", "Fair"
    undertone: str       # e.g., "Warm", "Cool", "Neutral"
    face_shape: str      # e.g., "Oval", "Round"
    eye_color: str
    confidence_score: int
    summary: str

# Product Recommendation Request
class ProductShade(BaseModel):
    id: str
    name: str            # e.g., "Peachy Blush"
    hex_code: str        # e.g., "#FFDAB9"
    category: str        # "Lipstick", "Foundation"

class RecommendationRequest(BaseModel):
    user_profile: SkinAnalysisResult
    products: List[ProductShade]

# Recommendation Response 
class MatchResult(BaseModel):
    best_match_id: str
    match_score: int     # 0-100%
    reasoning: str       # AI explanation