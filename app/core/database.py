# app/core/database.py — SQLAlchemy engine/session setup for the local SQLite store.
#
# This is the ONE database the backend has (users, chat_sessions, chat_messages).
# Kept deliberately simple: a single file-based SQLite DB, same "local, gitignored,
# dev-friendly" convention the old users.json used. Swappable later via DATABASE_URL.
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# check_same_thread=False: FastAPI can serve a request on a different thread than the
# one that created the connection pool; SQLite's default disallows cross-thread use of
# a single connection object, but SQLAlchemy's pooling + our per-request session
# lifecycle (one Session per request, closed at the end) make this safe.
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a request-scoped DB session, always closed after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all() -> None:
    """Create tables that don't exist yet. Safe to call repeatedly (no-op if present)."""
    # Import models so they're registered on Base.metadata before create_all runs.
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
