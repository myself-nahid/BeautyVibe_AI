"""
app/services/ai_service.py
──────────────────────────
All interactions with the OpenAI API live here.

Design principles
─────────────────
• One AsyncOpenAI client shared across the process lifetime (created once,
  reused everywhere) — avoids per-request connection overhead.
• Image bytes are validated (type + size) before hitting the API.
• Both functions raise typed HTTPExceptions on failure; callers do not need
  to handle raw OpenAI exceptions.
• A safe fallback is returned from analyze_face_image() if the AI call fails,
  so a broken API key never takes the whole endpoint down.
"""

import base64
import json
import logging
from typing import Any

from fastapi import HTTPException, status
from openai import AsyncOpenAI, OpenAIError

from app.core.config import get_settings
# from app.core.exceptions import AIServiceError
from app.core.config import Settings
from app.core.exceptions import AIServiceError
from app.schemas import MatchResult, ProductShade, SkinAnalysisResult

logger = logging.getLogger(__name__)
settings = get_settings()

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

_MIME_TO_PREFIX: dict[str, str] = {
    "image/jpeg": "data:image/jpeg;base64,",
    "image/jpg":  "data:image/jpeg;base64,",
    "image/png":  "data:image/png;base64,",
    "image/webp": "data:image/webp;base64,",
}

_FALLBACK_ANALYSIS: dict[str, Any] = {
    "skin_tone": "Medium",
    "undertone": "Neutral",
    "face_shape": "Oval",
    "eye_color": "Unknown",
    "confidence_score": 0,
    "summary": "Analysis unavailable — please try again.",
}


def _detect_mime(image_bytes: bytes) -> str:
    """
    Detect image MIME type from magic bytes.
    Returns a string like 'image/jpeg'.
    """
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] in (b"RIFF", b"WEBP") or b"WEBP" in image_bytes[:12]:
        return "image/webp"
    return "application/octet-stream"


def _validate_image(image_bytes: bytes) -> str:
    """
    Validate image size and type.
    Returns the detected MIME type on success.
    Raises HTTPException on failure.
    """
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
    """
    Return a valid face-shape string.
    Falls back to parsing the summary if the AI omitted the field.
    """
    shape = (result.get("face_shape") or "").strip().title()
    valid = {"Oval", "Round", "Square", "Heart", "Diamond", "Oblong"}

    if shape in valid:
        return shape

    # Secondary: scan the summary text
    summary = (result.get("summary") or "").lower()
    for candidate in ("oval", "round", "square", "heart", "diamond", "oblong"):
        if candidate in summary:
            return candidate.title()

    return "Oval"  # final safe default

async def analyze_face_image(image_bytes: bytes) -> dict[str, Any]:
    """
    Analyse a face image using GPT-4o vision.

    Returns a dict with keys:
        skin_tone, undertone, face_shape, eye_color,
        confidence_score, summary

    Never raises — returns _FALLBACK_ANALYSIS on any error so that
    the endpoint stays alive even if the AI call fails.
    """
    mime = _validate_image(image_bytes)   # may raise HTTPException
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
                {
                    "role": "system",
                    "content": "You are a beauty AI analyst. Output JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0.2,   # low temperature for consistent classifications
        )

        raw = response.choices[0].message.content
        result: dict[str, Any] = json.loads(raw)

        # Sanitise fields
        result["face_shape"] = _resolve_face_shape(result)
        result["confidence_score"] = max(0, min(100, int(result.get("confidence_score", 0))))
        result.setdefault("eye_color", "Unknown")
        result.setdefault("summary", "")

        logger.info(
            "Face analysis complete — tone=%s undertone=%s confidence=%s",
            result.get("skin_tone"),
            result.get("undertone"),
            result.get("confidence_score"),
        )
        return result

    except (OpenAIError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("analyze_face_image error: %s", exc)
        fallback = dict(_FALLBACK_ANALYSIS)
        fallback["summary"] = f"Analysis error: {exc}"
        return fallback


async def get_shade_recommendation(
    profile: SkinAnalysisResult,
    products: list[ProductShade],
) -> MatchResult:
    """
    Use GPT-4o to select the single best-matching product shade
    for the given beauty profile.

    Raises AIServiceError (HTTP 502) if the API call fails.
    """
    products_payload = [p.model_dump() for p in products]

    prompt = (
        "You are a Professional Color Theorist and Makeup Artist.\n\n"
        f"User Beauty Profile:\n"
        f"  Skin Tone : {profile.skin_tone}\n"
        f"  Undertone : {profile.undertone}\n"
        f"  Eye Color : {profile.eye_color}\n\n"
        "Available Products (JSON array):\n"
        f"{json.dumps(products_payload, indent=2)}\n\n"
        "Task:\n"
        "Select the SINGLE product from the list that best complements the "
        "user's skin tone and undertone based on colour theory.\n\n"
        "Return ONLY a JSON object with these exact keys:\n"
        "  best_match_id : the 'id' string of the chosen product\n"
        "  match_score   : integer 0-100 (100 = perfect match)\n"
        "  reasoning     : 1-2 sentences explaining the colour-theory rationale\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON — no markdown, no extra text.\n"
        "- best_match_id MUST be one of the ids in the product list."
    )

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a colour match expert. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2,
        )

        data: dict[str, Any] = json.loads(response.choices[0].message.content)

        # Validate the returned id actually exists in the supplied product list
        valid_ids = {p.id for p in products}
        if data.get("best_match_id") not in valid_ids:
            logger.warning(
                "AI returned unknown product id '%s'. Defaulting to first product.",
                data.get("best_match_id"),
            )
            data["best_match_id"] = products[0].id

        # Attach the full matched product details
        matched = next((p for p in products if p.id == data["best_match_id"]), None)

        logger.info(
            "Shade recommendation: id=%s score=%s",
            data["best_match_id"],
            data.get("match_score"),
        )

        return MatchResult(
            best_match_id=data["best_match_id"],
            match_score=max(0, min(100, int(data.get("match_score", 0)))),
            reasoning=data.get("reasoning", ""),
            matched_product=matched,
        )

    except (OpenAIError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.error("get_shade_recommendation error: %s", exc)
        raise AIServiceError(f"Recommendation failed: {exc}") from exc