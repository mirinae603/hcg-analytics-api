# app/models.py — SQLAlchemy ORM models for the local SQLite store.
#
# Three tables:
#   users          — accounts (signup/signin, role + approval status)
#   chat_sessions  — one row per AI Analyst conversation, tagged with its creator
#   chat_messages  — the turns within a session (user question / assistant reply)
#
# Role/status are plain strings (not a DB-level ENUM) — sqlite has no native enum type
# and the rest of this codebase already prefers loose string constants over DB enums
# (see the old UserStatus str-Enum in authentication_service.py). Valid values are
# enforced in the service layer.
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserRole:
    ADMIN = "admin"
    MEMBER = "member"


class UserStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default=UserRole.MEMBER)
    status = Column(String, nullable=False, default=UserStatus.PENDING)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    chat_sessions = relationship("ChatSession", back_populates="creator")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False, default="New Chat")
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    creator = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")
