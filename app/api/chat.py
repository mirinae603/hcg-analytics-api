# app/api/chat.py — AI Analyst chat session endpoints (all require a logged-in user).
#
# Visibility is deliberately open: ANY authenticated user can list, open, and post to
# ANY session, regardless of who created it — "chat is common", tagged with (not
# restricted by) creator. See app/services/chat_service.py for the persistence layer
# and app/ai/orchestrator.py for the actual AI-answering logic this wraps.
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.ai import page_context
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models import User
from app.services import chat_service

router = APIRouter()

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1)


# ---------- page context ----------
# What the user is looking at when they ask (Formulary or Spend page, which tab, month,
# hospital and filters), so "why is this low?" can be answered about THAT screen. It is
# accepted as any JSON value and reduced by app/ai/page_context.normalise to a fixed set of
# keys whose values come from fixed lists: the values end up inside a system message, and
# chat sessions are shared by every logged-in user, so no free text may pass. Not a pydantic
# model on purpose: a model either fails the request (a 422 for the whole question) or drops
# the whole hint over one value it did not expect, such as a new tab. normalise drops only
# what it does not know, key by key, and never fails the question.


class PostMessageRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # "fast" (default) = the single-pass orchestrator. "deep" = the reasoning swarm.
    # Defaulted so every existing caller, and the non-streaming route, are unaffected.
    mode: str = Field("fast", pattern="^(fast|deep)$")
    # Optional, so a client that does not send it (every caller before the page context
    # existed) behaves exactly as before. Read only through _context_dict.
    context: Any = None


def _context_dict(req: PostMessageRequest) -> Optional[Dict[str, Any]]:
    # None, not {}, when nothing valid is left: the engine is then called without the
    # keyword at all (see chat_service._answer_events).
    return page_context.normalise(req.context) or None


# Every route below except the stream is a plain `def`, not `async def`. They all do
# blocking work (SQLAlchemy, and for POST .../messages a whole AI turn of 10-60 s). In an
# `async def` that work runs ON the event loop, and while it does the single uvicorn
# process serves nothing else: dashboards, sign-in and other users' chats all stall.
# FastAPI runs a plain `def` route in its thread pool instead.
@router.post("/chat/sessions")
def create_chat_session(
    req: CreateSessionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    session = chat_service.create_session(db, user, req.title)
    return chat_service.get_session_detail(db, session.id)


@router.get("/chat/sessions")
def list_chat_sessions(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return {"sessions": chat_service.list_sessions(db)}


@router.get("/chat/sessions/{session_id}")
def get_chat_session(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return chat_service.get_session_detail(db, session_id)


@router.post("/chat/sessions/{session_id}/messages")
def post_chat_message(
    session_id: int,
    req: PostMessageRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return chat_service.post_message(db, session_id, req.query, context=_context_dict(req))


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    chat_service.delete_session(db, session_id)
    return {"deleted": True, "id": session_id}


@router.patch("/chat/sessions/{session_id}")
def rename_chat_session(
    session_id: int,
    req: RenameSessionRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    return chat_service.rename_session(db, session_id, req.title)


@router.post("/chat/sessions/{session_id}/messages/stream")
async def post_chat_message_stream(
    session_id: int,
    req: PostMessageRequest,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Live-progress counterpart to POST .../messages (above), which is left completely
    untouched. Validates eagerly -- same 404/400 semantics as every other route here --
    BEFORE constructing the StreamingResponse: a generator's body doesn't run until it
    is first iterated, i.e. after headers (a 200) have already gone out, so a bad
    session id or empty query must still fail with a normal REST status here, not a 200
    stream that then carries an error event.

    This one stays `async def`: its own body is two short lookups, and Starlette iterates
    the sync generator below in the thread pool, so the AI turn never blocks the loop."""
    chat_service.get_session_or_404(db, session_id)
    if not (req.query or "").strip():
        raise HTTPException(status_code=400, detail="Message text is required")

    context = _context_dict(req)

    def gen():
        for ev in chat_service.stream_message_events(db, session_id, req.query, mode=req.mode,
                                                     context=context):
            yield f"data: {json.dumps(ev)}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                       "X-Accel-Buffering": "no"})
