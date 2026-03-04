"""
app/core/security.py
────────────────────
Validates that the caller (Main Backend) is authorized.
"""
from fastapi import Header, HTTPException, status
from app.core.config import get_settings

settings = get_settings()

async def verify_api_key(x_api_key: str = Header(..., description="Internal Service Secret")):
    """
    Dependency that checks the X-API-KEY header.
    If it doesn't match the .env value, reject the request.
    """
    if x_api_key != settings.AI_SERVICE_SECRET:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    return x_api_key