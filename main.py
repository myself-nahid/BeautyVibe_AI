"""
main.py
───────
GlowFlow AI Backend — FastAPI application entry point.

Startup sequence
────────────────
1. Load settings from .env (validated by pydantic-settings).
2. Configure structured logging.
3. Create database tables via the lifespan hook.
4. Register CORS middleware.
5. Mount the v1 API router.
6. Register a global exception handler for clean error responses.
"""

import logging
import logging.config
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes
from app.core.config import get_settings
from app.database import create_db_and_tables
from app.schemas import HealthResponse

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
        # Quieten SQLAlchemy in production
        "loggers": {
            "sqlalchemy.engine": {
                "level": "DEBUG" if get_settings().APP_DEBUG else "WARNING",
                "handlers": ["console"],
                "propagate": False,
            }
        },
    }
)

logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup / shutdown logic around the application lifetime."""
    logger.info("Starting GlowFlow backend (env=%s)", settings.APP_ENV)
    create_db_and_tables()
    logger.info("Database tables created / verified.")
    yield
    logger.info("GlowFlow backend shutting down.")

app = FastAPI(
    title="GlowFlow AI API",
    version="1.0.0",
    description=(
        "AI-powered beauty commerce backend.\n\n"
        "Core features:\n"
        "- **Face analysis** — skin tone, undertone, face shape, eye colour\n"
        "- **Shade recommendation** — colour-theory product matching\n"
        "- **Profile history** — persisted beauty profiles per user\n"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler so unhandled exceptions return a clean JSON response
    instead of an HTML traceback (important in production).
    """
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )

app.include_router(routes.router, prefix="/api/v1")

@app.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe",
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="GlowFlow Backend is Running 🚀",
        version="1.0.0",
        environment=settings.APP_ENV,
    )