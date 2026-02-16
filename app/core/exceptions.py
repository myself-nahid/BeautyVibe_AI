"""
app/core/exceptions.py
──────────────────────
Custom application exceptions with HTTP status mappings.
Raised inside services/routes; caught and formatted by the
global exception handler registered in main.py.
"""

from fastapi import HTTPException, status


class ImageTooLargeError(HTTPException):
    def __init__(self, max_mb: int) -> None:
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {max_mb} MB size limit.",
        )


class UnsupportedImageTypeError(HTTPException):
    def __init__(self, content_type: str) -> None:
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{content_type}'. "
                "Accepted: JPEG, PNG, WebP."
            ),
        )


class AIServiceError(HTTPException):
    def __init__(self, detail: str = "AI service encountered an error.") -> None:
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        )


class ProductNotFoundError(HTTPException):
    def __init__(self, product_id: str) -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with id '{product_id}' was not found in the provided list.",
        )