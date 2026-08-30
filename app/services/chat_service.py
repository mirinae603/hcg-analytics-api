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
from sqlalchemy import update as _sa_update
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
    just the text back out for conversational context).

    Also re-embeds the "[active scope: …]" marker orchestrator._latest_scope() looks
    for. That marker was designed for the OLD ephemeral /ai/chat flow, where the
    FRONTEND appended it to the content it echoed back as history on the next turn.
    The session-backed path never had frontend echoing at all — history is rebuilt
    here, straight from the DB — so without this the marker could never appear and
    orchestrator.answer() always saw prior_scope="" on every follow-up. That's not
    hypothetical: it was caught live — "how does that compare to last month?" right
    after an inventory-turnover-ratio answer came back discussing procurement spend
    instead, because the model had zero idea what "that" referred to. `scope` is
    already computed and yielded by orchestrator.answer() on every "answer" event (see
    orchestrator.py's `scope = ...` line just above `yield {"type": "answer", ...}`);
    the bug was purely that this file discarded it instead of storing it."""
    history: List[Dict] = []
    for m in session.messages:
        if m.role == "user":
            history.append({"role": "user", "content": m.content})
        else:
            try:
                parsed = json.loads(m.content)
                text = parsed.get("text", "") if isinstance(parsed, dict) else str(parsed)
                scope = parsed.get("scope") if isinstance(parsed, dict) else None
            except (TypeError, ValueError):
                text, scope = m.content, None
            if scope:
                text = f"{text}\n\n[active scope: {scope}]"
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
    scope: Optional[str] = None
    try:
        for ev in orchestrator.answer(query, history):
            t = ev.get("type")
            if t == "answer":
                text = ev.get("text")
                verified = ev.get("verified")
                options = ev.get("options") or options
                # See _history_for_orchestrator's docstring: dropping this is exactly what
                # broke multi-turn follow-ups (every prior scope silently reset to "").
                scope = ev.get("scope") or scope
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

    # Empty counts as missing, not just None. A turn once persisted `"text": ""` with a
    # full chart, table, badge and follow-up chips attached — the UI faithfully rendered
    # every one of those around a blank answer, which reads as the product being broken.
    # Whatever the cause upstream, a reply with no words is never worth storing as one.
    if text is None or not str(text).strip():
        text = "I couldn't generate a response for that."
        kind = "error"

    return {"text": text, "verified": verified, "options": options, "chart": chart,
            "table": table, "queries": queries, "kind": kind, "scope": scope}


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


def delete_session(db: Session, session_id: int) -> None:
    """Hard delete. `cascade="all, delete-orphan"` on ChatSession.messages (models.py)
    means db.delete(session) removes every message too — verified via the ORM object,
    not a raw bulk DELETE, which is what actually triggers that cascade.

    "Chat is common" (see this file's module docstring): the same design that lets any
    signed-in user list/open/post to any session also lets any signed-in user delete
    any session, not just their own. Not a gap — a deliberate extension of the existing
    visibility model already covered by test_admin_can_see_a_members_session and
    test_different_logged_in_user_can_list_and_see_who_created_session."""
    session = get_session_or_404(db, session_id)
    db.delete(session)
    db.commit()


def rename_session(db: Session, session_id: int, title: str) -> Dict:
    session = get_session_or_404(db, session_id)
    clean = (title or "").strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Title is required")
    # A rename isn't new conversation activity, and bumping updated_at would reorder the
    # sidebar (jump the session to "Today") purely from a title edit. The column has
    # onupdate=_utcnow (models.py), which fires whenever the row is UPDATEd and that
    # column has no explicit value in the SAME statement. Re-assigning `session.updated_at
    # = session.updated_at` through the ORM does NOT count as "explicit" for this purpose
    # -- with nothing changed, SQLAlchemy's dirty-tracking leaves it out of the SET clause
    # entirely, so onupdate still wins. (Caught by test_rename_does_not_reorder_updated_at,
    # which asserts on the actual timestamp, not just that the title changed -- the first
    # version of this fix looked right and still silently failed it.) A Core-level UPDATE
    # naming both columns explicitly is unambiguous: onupdate only fires for a column
    # genuinely absent from .values().
    original_updated_at = session.updated_at
    db.execute(
        _sa_update(ChatSession)
        .where(ChatSession.id == session_id)
        .values(title=clean[:200], updated_at=original_updated_at)
    )
    db.commit()
    db.refresh(session)
    return {"id": session.id, "title": session.title}


def stream_message_events(db: Session, session_id: int, query: str, mode: str = "fast"):
    """Generator of SSE-shaped event dicts for a session-backed turn — the live-progress
    counterpart to post_message(). Reuses orchestrator.answer() exactly as post_message
    does (same history, same audit gate), but yields step/sql/answer/chart/table/done AS
    THEY HAPPEN instead of consolidating them into one blocking response, then persists
    the same consolidated shape post_message would have stored, once, after "done".

    Deliberately ADDITIVE: POST /chat/sessions/{id}/messages (post_message, above) is
    left completely untouched, so none of the 12 existing tests in test_chat.py change
    behaviour. This is a new route the frontend opts into; the old one stays a working,
    tested fallback.

    The user's message is persisted FIRST, before any streaming starts — so if the
    client disconnects mid-answer (closed tab, network drop), the question they asked
    is never lost, only the reply. Same durability guarantee post_message already gives
    the user side; this extends it to the reply side by accumulating exactly what
    _run_ai_turn accumulates and writing it once the stream concludes.
    """
    session = get_session_or_404(db, session_id)
    query = (query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message text is required")

    # The ONLY place the two answer paths meet. `deep` is a separate package with its own
    # client, its own loop and its own prompts; it exposes the same generator contract, so
    # nothing below this line — accumulation, persistence, the event vocabulary — differs
    # between modes, and nothing in deep mode can regress the fast path.
    engine = orchestrator
    if str(mode).lower() == "deep":
        from app.ai.deep import engine as engine  # noqa: PLC0415

    history = _history_for_orchestrator(session)

    user_msg = ChatMessage(session_id=session.id, role="user", content=query)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    yield {"type": "user_message", "message": _serialize_message(user_msg)}

    text: Optional[str] = None
    verified: Optional[str] = None
    options: List[str] = []
    chart = None
    table = None
    queries: List[Dict] = []
    kind = "answer"
    scope: Optional[str] = None

    try:
        for ev in engine.answer(query, history):
            t = ev.get("type")
            if t == "answer":
                text = ev.get("text"); verified = ev.get("verified")
                options = ev.get("options") or options
                scope = ev.get("scope") or scope
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
                text = ev.get("text"); options = ev.get("options") or options; kind = "clarify"
            elif t == "error":
                text = ev.get("text"); kind = "error"
            # Forward every event live — "step"/"done" included, so the UI can show real
            # progress instead of a single frozen spinner label for the whole turn.
            yield ev
    except Exception as e:  # the orchestrator/Azure call itself blew up mid-stream
        text = f"The AI Analyst hit an error answering that: {e}"
        kind = "error"
        yield {"type": "error", "text": text}

    if text is None:
        text = "I couldn't generate a response for that."
        kind = "error"

    reply = {"text": text, "verified": verified, "options": options, "chart": chart,
             "table": table, "queries": queries, "kind": kind, "scope": scope}

    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=json.dumps(reply))
    db.add(assistant_msg)
    session.updated_at = datetime.now(timezone.utc)
    db.add(session)
    db.commit()
    db.refresh(assistant_msg)

    yield {"type": "persisted", "assistant_message": _serialize_message(assistant_msg)}


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
