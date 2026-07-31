# app/api/chat.py — AI Analyst chat session endpoints (all require a logged-in user).
#
# Visibility is deliberately open: ANY authenticated user can list, open, and post to
# ANY session, regardless of who created it — "chat is common", tagged with (not
# restricted by) creator. See app/services/chat_service.py for the persistence layer
# and app/ai/orchestrator.py for the actual AI-answering logic this wraps.
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models import User
from app.services import chat_service

router = APIRouter()


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class PostMessageRequest(BaseModel):
    query: str = Field(..., min_length=1)


@router.post("/chat/sessions")
async def create_chat_session(
    req: CreateSessionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = chat_service.create_session(db, user, req.title)
    return chat_service.get_session_detail(db, session.id)


@router.get("/chat/sessions")
async def list_chat_sessions(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return {"sessions": chat_service.list_sessions(db)}


@router.get("/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return chat_service.get_session_detail(db, session_id)


@router.post("/chat/sessions/{session_id}/messages")
async def post_chat_message(
    session_id: int,
    req: PostMessageRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return chat_service.post_message(db, session_id, req.query)
