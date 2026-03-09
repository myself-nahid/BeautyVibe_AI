"""
app/services/ai_service.py
──────────────────────────
All interactions with the OpenAI API live here.
"""

import base64
import json
import logging
from typing import Any

from fastapi import HTTPException, status
from openai import AsyncOpenAI, OpenAIError

from app.core.config import get_settings
from app.core.exceptions import AIServiceError
from app.schemas import (
    Category,
    MatchedProduct,
    MatchResult,
    ProductShade,
    RecommendationResponse,
    SkinAnalysisResult,
)

logger   = logging.getLogger(__name__)
settings = get_settings()

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

_MIME_TO_PREFIX: dict[str, str] = {
    "image/jpeg": "data:image/jpeg;base64,",
    "image/jpg":  "data:image/jpeg;base64,",
    "image/png":  "data:image/png;base64,",
    "image/webp": "data:image/webp;base64,",
}

_FALLBACK_ANALYSIS: dict[str, Any] = {
    "skin_tone":        "Medium",
    "undertone":        "Neutral",
    "face_shape":       "Oval",
    "eye_color":        "Unknown",
    "confidence_score": 0,
    "summary":          "Analysis unavailable — please try again.",
}


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def _detect_mime(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] in (b"RIFF", b"WEBP") or b"WEBP" in image_bytes[:12]:
        return "image/webp"
    return "application/octet-stream"


def _validate_image(image_bytes: bytes) -> str:
    if len(image_bytes) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.MAX_IMAGE_SIZE_MB} MB limit.",
        )
    mime = _detect_mime(image_bytes)
    if mime not in _MIME_TO_PREFIX:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image type '{mime}'. Accepted: JPEG, PNG, WebP.",
        )
    return mime


def _resolve_face_shape(result: dict[str, Any]) -> str:
    shape = (result.get("face_shape") or "").strip().title()
    valid = {"Oval", "Round", "Square", "Heart", "Diamond", "Oblong"}
    if shape in valid:
        return shape
    summary = (result.get("summary") or "").lower()
    for candidate in ("oval", "round", "square", "heart", "diamond", "oblong"):
        if candidate in summary:
            return candidate.title()
    return "Oval"


def _build_product_payload(products: list[ProductShade]) -> list[dict]:
    """Slim product list for the AI prompt — colour-relevant fields only."""
    payload = []
    for p in products:
        entry: dict[str, Any] = {"id": p.id, "name": p.name}
        if p.shade:    entry["shade"]    = p.shade
        if p.hex_code: entry["hex_code"] = p.hex_code
        if p.brand:    entry["brand"]    = p.brand
        if p.category: entry["category"] = p.category
        payload.append(entry)
    return payload


# ─────────────────────────────────────────────
# Face analysis
# ─────────────────────────────────────────────

async def analyze_face_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Analyse a face image using GPT-4o vision.
    Never raises — returns _FALLBACK_ANALYSIS on any error.
    """
    mime     = _validate_image(image_bytes)
    data_url = _MIME_TO_PREFIX[mime] + base64.b64encode(image_bytes).decode()

    prompt = (
        "You are a professional beauty consultant with expertise in skin analysis.\n\n"
        "Carefully analyse the face in this image and return ONLY a JSON object "
        "with these exact keys:\n"
        "  skin_tone        : one of [Fair, Light, Medium, Tan, Deep]\n"
        "  undertone        : one of [Cool, Neutral, Warm]\n"
        "  face_shape       : one of [Oval, Round, Square, Heart, Diamond, Oblong]\n"
        "  eye_color        : descriptive string (e.g. 'Dark Brown', 'Hazel')\n"
        "  confidence_score : integer 0-100 representing your confidence\n"
        "  summary          : one sentence describing the person's key features\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON — no markdown, no explanation.\n"
        "- Never set any field to null; always choose the closest match.\n"
        "- confidence_score must reflect how clearly the features are visible."
    )

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a beauty AI analyst. Output JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",      "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.2,
        )

        raw    = response.choices[0].message.content
        result: dict[str, Any] = json.loads(raw)

        result["face_shape"]       = _resolve_face_shape(result)
        result["confidence_score"] = max(0, min(100, int(result.get("confidence_score", 0))))
        result.setdefault("eye_color", "Unknown")
        result.setdefault("summary",   "")

        logger.info(
            "Face analysis complete — tone=%s undertone=%s confidence=%s",
            result.get("skin_tone"), result.get("undertone"), result.get("confidence_score"),
        )
        return result

    except (OpenAIError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("analyze_face_image error: %s", exc)
        fallback = dict(_FALLBACK_ANALYSIS)
        fallback["summary"] = f"Analysis error: {exc}"
        return fallback


# ─────────────────────────────────────────────
# Shade recommendations
# ─────────────────────────────────────────────

async def get_shade_recommendations(
    profile:    SkinAnalysisResult,
    products:   list[ProductShade],
    categories: list[Category],
    top_n:      int = 3,
) -> RecommendationResponse:
    """
    Use GPT-4o to return top-N ranked shade recommendations.

    - categories are passed through to the response as a separate list.
    - matched_product is built from the original input product only —
      the AI never reconstructs product data.

    Raises AIServiceError (HTTP 502) on failure.
    """
    actual_top_n     = min(top_n, len(products))
    product_map      = {p.id: p for p in products}
    products_payload = _build_product_payload(products)

    prompt = (
        "You are a Professional Color Theorist and Makeup Artist.\n\n"
        "## User Beauty Profile\n"
        f"  Skin Tone : {profile.skin_tone}\n"
        f"  Undertone : {profile.undertone}\n"
        f"  Eye Color : {profile.eye_color}\n\n"
        "## Available Products\n"
        f"{json.dumps(products_payload, indent=2)}\n\n"
        "## Task\n"
        f"Select the TOP {actual_top_n} products that best complement the user's "
        "skin tone and undertone. Rank from best match to worst.\n\n"
        "Colour-theory rules:\n"
        "- Warm undertones → earthy, coral, terracotta, bronze, warm nude shades.\n"
        "- Cool undertones → berry, mauve, rose, burgundy, cool nude shades.\n"
        "- Neutral undertones → most shades work; prefer balanced nudes.\n"
        "- Deep skin tones → rich, highly pigmented shades show up best.\n"
        "- Use shade name and hex_code as the primary colour signal.\n\n"
        "## Output — return ONLY this JSON, no markdown:\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        '      "best_match_id": "<id string exactly as given>",\n'
        '      "match_score":   <integer 0-100>,\n'
        '      "reasoning":     "<1-2 sentences citing shade name and colour theory>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- best_match_id MUST exactly match one of the id values above.\n"
        "- No duplicate ids.\n"
        f"- matches array must have exactly {actual_top_n} entries."
    )

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a colour match expert. Output JSON only."},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.2,
        )

        data: dict[str, Any] = json.loads(response.choices[0].message.content)
        matches_raw: list[dict] = data.get("matches", [])

        valid_ids = set(product_map.keys())
        seen_ids: set[str]          = set()
        results:  list[MatchResult] = []

        for item in matches_raw:
            match_id = str(item.get("best_match_id", "")).strip()

            if match_id not in valid_ids:
                logger.warning("AI returned unknown product id '%s'. Skipping.", match_id)
                continue
            if match_id in seen_ids:
                logger.warning("AI returned duplicate product id '%s'. Skipping.", match_id)
                continue

            seen_ids.add(match_id)
            original = product_map[match_id]

            results.append(
                MatchResult(
                    best_match_id=match_id,
                    match_score=max(0, min(100, int(item.get("match_score", 0)))),
                    reasoning=item.get("reasoning", ""),
                    # ✅ Built from original input — no AI reconstruction
                    matched_product=MatchedProduct.from_product_shade(original),
                )
            )

        # Pad if AI returned fewer than requested
        if len(results) < actual_top_n:
            logger.warning("AI returned %d/%d — padding.", len(results), actual_top_n)
            for product in products:
                if product.id not in seen_ids:
                    results.append(
                        MatchResult(
                            best_match_id=product.id,
                            match_score=0,
                            reasoning="Fallback entry — insufficient AI results.",
                            matched_product=MatchedProduct.from_product_shade(product),
                        )
                    )
                    seen_ids.add(product.id)
                    if len(results) >= actual_top_n:
                        break

        final = results[:actual_top_n]
        logger.info(
            "Recommendations done — returned=%d profile=%s/%s",
            len(final), profile.skin_tone, profile.undertone,
        )

        return RecommendationResponse(
            categories=categories,          # ✅ separate — never inside matched_product
            recommendations=final,
            total=len(final),
        )

    except (OpenAIError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("get_shade_recommendations error: %s", exc)
        raise AIServiceError(f"Recommendation failed: {exc}") from exc