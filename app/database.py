"""
app/database.py
───────────────
Database engine + session factory using SQLModel.

• Engine is created once at import time from Settings.
• get_session() is a FastAPI dependency that yields a managed
  session (auto-commit / rollback on error, always closed).
• create_db_and_tables() is called from the lifespan hook in main.py.
"""

from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings

settings = get_settings()

# SQLite needs check_same_thread=False; other drivers do not.
_connect_args = {"check_same_thread": False} if settings.is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,       # SQL logging only in debug mode
    connect_args=_connect_args,
)


def create_db_and_tables() -> None:
    """Create all tables declared in SQLModel models."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency — yields a database session.
    The 'with' block ensures the session is always closed and any
    uncommitted transaction is rolled back on exception.
    """
    with Session(engine) as session:
        yield session