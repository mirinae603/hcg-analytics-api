# app/services/chat_service.py — AI Analyst chat persistence.
#
# "Chat is common": every session and message is tagged with its creator, but ANY
# authenticated user can list/open/post to ANY session — see the module-level design
# note in the task brief. This file owns exactly that persistence; it does NOT
# reimplement AI-answering logic — `_run_ai_turn` calls the very same
# app.ai.orchestrator.answer() generator that POST /ai/chat already streams from, and
# only consolidates its SSE-shaped events into one stored reply.
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.ai import orchestrator
from app.models import ChatMessage, ChatSession, User


def _user_public(user: Optional[User]) -> Optional[Dict]:
    if user is None:
        return None
    name = f"{user.first_name} {user.last_name}".strip() or user.email
    return {"id": user.id, "name": name, "email": user.email}


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _serialize_message(m: ChatMessage) -> Dict:
    if m.role == "assistant":
        try:
            content = json.loads(m.content)
        except (TypeError, ValueError):
            content = {"text": m.content}
    else:
        content = m.content
    return {
        "id": m.id,
        "role": m.role,
        "content": content,
        "created_at": _iso(m.created_at),
    }


def _history_for_orchestrator(session: ChatSession) -> List[Dict]:
    """Rebuild the plain {role, content} history app.ai.orchestrator.answer expects
    from what's actually stored (assistant turns are stored as a JSON envelope, so pull
    just the text back out for conversational context)."""
    history: List[Dict] = []
    for m in session.messages:
        if m.role == "user":
            history.append({"role": "user", "content": m.content})
        else:
            try:
                parsed = json.loads(m.content)
                text = parsed.get("text", "") if isinstance(parsed, dict) else str(parsed)
            except (TypeError, ValueError):
                text = m.content
            history.append({"role": "assistant", "content": text})
    return history


def _run_ai_turn(query: str, history: List[Dict]) -> Dict:
    """Consume orchestrator.answer's SSE-event generator into one consolidated reply.
    Never raises — any failure becomes an error-flavoured reply so a message is always
    persisted (and returned) instead of leaving a user turn with no response at all.

    `verified` is stored as the orchestrator's own "ok" | "corrected" | "flagged" | None
    string, NOT collapsed to a bool — Python truthy-strings mean bool("flagged") is True,
    which previously rendered a flagged (unreliable) answer with a green "Verified" badge,
    the opposite of what the badge means. Mirrors exactly what the live SSE /ai/chat path
    has always sent the frontend (see the "answer" event in app/ai/orchestrator.py)."""
    text: Optional[str] = None
    verified: Optional[str] = None
    options: List[str] = []
    chart = None
    table = None
    queries: List[Dict] = []
    kind = "answer"
    try:
        for ev in orchestrator.answer(query, history):
            t = ev.get("type")
            if t == "answer":
                text = ev.get("text")
                verified = ev.get("verified")
                options = ev.get("options") or options
            elif t == "chart":
                chart = ev.get("plotly")
            elif t == "table":
                tbl = ev.get("table")
                table = {**tbl, "note": ev.get("note", "")} if tbl else None
            elif t == "sql":
                queries.append({"purpose": ev.get("purpose"), "sql": ev.get("sql"),
                                 "rows": ev.get("rows"), "error": ev.get("error")})
            elif t == "followups":
                options = ev.get("options") or options
            elif t == "clarify":
                text = ev.get("text")
                options = ev.get("options") or options
                kind = "clarify"
            elif t == "error":
                text = ev.get("text")
                kind = "error"
            # "step" / "done" are progress-only — nothing to persist from them.
    except Exception as e:  # the orchestrator/Azure call itself blew up
        text = f"The AI Analyst hit an error answering that: {e}"
        kind = "error"

    if text is None:
        text = "I couldn't generate a response for that."
        kind = "error"

    return {"text": text, "verified": verified, "options": options, "chart": chart,
            "table": table, "queries": queries, "kind": kind}


# ---------- public API ----------

def create_session(db: Session, creator: User, title: Optional[str] = None) -> ChatSession:
    clean_title = (title or "").strip() or "New Chat"
    session = ChatSession(created_by=creator.id, title=clean_title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def list_sessions(db: Session) -> List[Dict]:
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    out = []
    for s in sessions:
        creator = db.query(User).filter(User.id == s.created_by).first()
        out.append({
            "id": s.id,
            "title": s.title,
            "created_at": _iso(s.created_at),
            "updated_at": _iso(s.updated_at),
            "created_by": _user_public(creator),
            "message_count": len(s.messages),
        })
    return out


def get_session_or_404(db: Session, session_id: int) -> ChatSession:
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


def get_session_detail(db: Session, session_id: int) -> Dict:
    session = get_session_or_404(db, session_id)
    creator = db.query(User).filter(User.id == session.created_by).first()
    return {
        "id": session.id,
        "title": session.title,
        "created_at": _iso(session.created_at),
        "updated_at": _iso(session.updated_at),
        "created_by": _user_public(creator),
        "messages": [_serialize_message(m) for m in session.messages],
    }


def post_message(db: Session, session_id: int, query: str) -> Dict:
    session = get_session_or_404(db, session_id)
    query = (query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message text is required")

    history = _history_for_orchestrator(session)

    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    db.add(user_msg)
    db.flush()  # assigns an id within the same transaction

    reply = _run_ai_turn(query, history)

    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=json.dumps(reply))
    db.add(assistant_msg)

    session.updated_at = datetime.now(timezone.utc)
    db.add(session)

    db.commit()
    db.refresh(user_msg)
    db.refresh(assistant_msg)

    return {
        "user_message": _serialize_message(user_msg),
        "assistant_message": _serialize_message(assistant_msg),
    }
