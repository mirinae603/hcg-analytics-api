# app/core/database.py — SQLAlchemy engine/session setup for the local SQLite store.
#
# This is the ONE database the backend has (users, chat_sessions, chat_messages).
# Kept deliberately simple: a single file-based SQLite DB, same "local, gitignored,
# dev-friendly" convention the old users.json used. Swappable later via DATABASE_URL.
from __future__ import annotations

import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

log = logging.getLogger(__name__)

# Environment variables that say the process runs in a container whose disk is thrown
# away on every restart, deploy or scale event (Azure Container Apps sets the first;
# a plain Docker/Kubernetes run is told apart by the second, which the image does not set).
EPHEMERAL_HOST_VARS = ("CONTAINER_APP_NAME", "KUBERNETES_SERVICE_HOST")


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def engine_options(url: str) -> dict:
    """create_engine keyword arguments for this URL.

    SQLite: check_same_thread=False. FastAPI can serve a request on a different thread
    than the one that created the connection pool; SQLite's default disallows cross-thread
    use of a single connection object, but SQLAlchemy's pooling + our per-request session
    lifecycle (one Session per request, closed at the end) make this safe.
    A server database (Postgres): pool_pre_ping, because a managed server closes idle
    connections and the first request after a quiet spell would otherwise fail on a dead
    one; the ping replaces it silently."""
    if is_sqlite(url):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True}


def ephemeral_storage_warning(url: str, environ=os.environ) -> str | None:
    """The warning to log when users and chats would live in a SQLite file on a container's
    throwaway disk: every deploy or restart then starts from an empty database, so every
    AI Analyst chat (and every signed-up user but the seeded admin) disappears. None when
    the URL is a server database or the process is not in such a container."""
    if not is_sqlite(url) or not any(environ.get(v) for v in EPHEMERAL_HOST_VARS):
        return None
    return ("DATABASE_URL is a SQLite file on this container's own disk, which is wiped on "
            "every deploy, restart and scale event: AI chat sessions and signed-up users will "
            "be lost. Set DATABASE_URL to a server database (postgresql+psycopg://...); see "
            "RUNBOOK_HCG.md 'Keep AI chats and users across deploys'.")


_warning = ephemeral_storage_warning(settings.DATABASE_URL)
if _warning:
    log.warning(_warning)

engine = create_engine(settings.DATABASE_URL, **engine_options(settings.DATABASE_URL))
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
