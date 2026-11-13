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
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy import update as _sa_update
from sqlalchemy.orm import Session

from app.ai import orchestrator
from app.models import ChatMessage, ChatSession, User

_log = logging.getLogger(__name__)
# The app configures no logging (as in app/ai/orchestrator.py): a plain stderr handler only when
# nobody has configured one, so a failed turn's traceback and its ref are seen.
if not _log.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)


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


def _history_limit() -> int:
    # Read at call time, so a change to the orchestrator's window is picked up here too.
    # Deep mode reads fewer turns than this (the last 6), so one bound serves both modes.
    return int(getattr(orchestrator, "HISTORY_MESSAGES", 24))


def _history_for_orchestrator(db: Session, session: ChatSession) -> List[Dict]:
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
    the bug was purely that this file discarded it instead of storing it.

    Only the last orchestrator.HISTORY_MESSAGES messages are read, newest first from the
    database and then put back in order. The orchestrator never looks further back, and
    sessions are shared by every user, so they only grow: loading `session.messages`
    pulled and JSON-parsed every stored reply (each with its chart and up to 50 table
    rows) on every turn, to throw nearly all of it away."""
    rows = (db.query(ChatMessage)
            .filter(ChatMessage.session_id == session.id)
            .order_by(ChatMessage.id.desc())
            .limit(_history_limit())
            .all())[::-1]
    history: List[Dict] = []
    for m in rows:
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


def _answer_events(engine, query: str, history: List[Dict], context: Optional[Dict[str, Any]]):
    """engine.answer(...) with the page context passed only when there is one, so an
    answer(query, history) that predates the keyword (and every test fake written that
    way) keeps working for a client that sends no context."""
    if context:
        return engine.answer(query, history, context=context)
    return engine.answer(query, history)


# What the user reads when a turn fails. Never the exception's own text: that can carry SQL,
# a file path or a service's error body. The ref is written to the server log beside the
# traceback, so support can find what happened from what the user saw.
_FAILED = "Sorry — something went wrong answering that. Please try again. (ref {ref})"
_PARTIAL = "Part of this answer couldn't be completed, so a chart or table may be missing. (ref {ref})"


def _has_text(text) -> bool:
    return text is not None and bool(str(text).strip())


class TurnReply:
    """One AI turn, folded from the engine's events into the reply that is stored. Both chat
    routes and POST /ai/chat use it, so the same events give the same reply and the same
    error events everywhere.

    `add(ev)` takes each event and returns the one to send on, `fail()` is called from the
    except block when the engine raises, `finish()` when it stops, and `reply()` gives the
    stored shape. A failure before the answer turns the turn into an error reply: no
    verified, no chart, no table, so no trust badge sits on it. A failure after the answer
    keeps everything the user already has — text, verified, charts, table — and adds a
    `notice`: the answer used to be replaced by the error while `verified` stayed
    "canonical", and the UI showed the green badge on an error.

    `verified` is stored as the orchestrator's own "ok" | "corrected" | "flagged" |
    "canonical" | None string, NOT collapsed to a bool — Python truthy-strings mean
    bool("flagged") is True, which previously rendered a flagged (unreliable) answer with a
    green "Verified" badge, the opposite of what the badge means."""

    def __init__(self, route: str):
        self.route = route
        self.text: Optional[str] = None
        self.verified: Optional[str] = None
        self.options: List[str] = []
        self.charts: List[Dict] = []
        self.table: Optional[Dict] = None
        self.queries: List[Dict] = []
        self.kind = "answer"
        self.scope: Optional[str] = None
        self.notice: Optional[str] = None
        # True once an answer with words has been sent. An empty answer is no answer: the
        # user has nothing to keep, so a failure after it is a failure before an answer.
        self.answered = False

    def add(self, ev: Dict) -> Dict:
        t = ev.get("type")
        if t == "answer":
            self.text = ev.get("text")
            self.verified = ev.get("verified")
            self.options = ev.get("options") or self.options
            # See _history_for_orchestrator's docstring: dropping this is exactly what
            # broke multi-turn follow-ups (every prior scope silently reset to "").
            self.scope = ev.get("scope") or self.scope
            self.answered = _has_text(self.text)
        elif t == "chart":
            if ev.get("plotly"):
                self.charts.append(ev["plotly"])
        elif t == "table":
            tbl = ev.get("table")
            self.table = {**tbl, "note": ev.get("note", "")} if tbl else None
        elif t == "sql":
            self.queries.append({"purpose": ev.get("purpose"), "sql": ev.get("sql"),
                                 "rows": ev.get("rows"), "error": ev.get("error")})
        elif t == "followups":
            self.options = ev.get("options") or self.options
        elif t == "clarify":
            self.text = ev.get("text")
            self.options = ev.get("options") or self.options
            self.kind = "clarify"
        elif t == "error":
            # The engine's own ref when it sent one (it logged the traceback under it), so the
            # ref the user is shown finds that traceback.
            ref = ev.get("ref") or secrets.token_hex(4)
            if self.answered:
                _log.error("chat.turn_error_after_answer route=%s ref=%s text=%s",
                           self.route, ref, ev.get("text"))
                return self._partial(ref)
            # Before an answer the engine's own error text is the reply: it is written for
            # the user ("isn't configured", "the AI service rejected the key"). One with no
            # words was sent on as it was, and finish() then sent the generic error as well:
            # two errors for one failure. It gets the generic words here, so it is the only one.
            text = ev.get("text")
            if not _has_text(text):
                _log.error("chat.turn_error_no_text route=%s ref=%s", self.route, ref)
                text = _FAILED.format(ref=ref)
            return self._error(text)
        # "step" / "done" are progress-only — nothing to persist from them.
        return ev

    def fail(self) -> Dict:
        """The error event for an exception the engine raised. Call it from the except block,
        so the traceback is logged with the ref the user is shown."""
        ref = secrets.token_hex(4)
        _log.exception("chat.turn_failed route=%s ref=%s answered=%s", self.route, ref, self.answered)
        if self.answered:
            return self._partial(ref)
        return self._error(_FAILED.format(ref=ref))

    def finish(self) -> Optional[Dict]:
        """Close the turn. A turn that ends with no words is a failed turn: a reply once
        persisted `"text": ""` with a chart, a table, a badge and follow-up chips around it,
        which reads as the product being broken. Returns the error event to send then, else
        None."""
        if _has_text(self.text):
            return None
        ref = secrets.token_hex(4)
        _log.error("chat.turn_no_text route=%s ref=%s", self.route, ref)
        return self._error(_FAILED.format(ref=ref))

    def reply(self) -> Dict:
        out = {"text": self.text, "verified": self.verified, "options": self.options,
               # "chart" (the last one) stays for readers older than "charts".
               "chart": self.charts[-1] if self.charts else None, "charts": self.charts,
               "table": self.table, "queries": self.queries, "kind": self.kind, "scope": self.scope}
        if self.notice:
            out["notice"] = self.notice
        return out

    def _partial(self, ref: str) -> Dict:
        self.notice = _PARTIAL.format(ref=ref)
        return {"type": "error", "text": self.notice, "partial": True}

    def _error(self, text: Optional[str]) -> Dict:
        self.text = text
        self.kind = "error"
        self.verified = None
        self.options = []
        self.charts = []
        self.table = None
        return {"type": "error", "text": text}


def _run_ai_turn(query: str, history: List[Dict], context: Optional[Dict[str, Any]] = None) -> Dict:
    """Consume orchestrator.answer's SSE-event generator into one consolidated reply.
    Never raises — any failure becomes an error-flavoured reply so a message is always
    persisted (and returned) instead of leaving a user turn with no response at all."""
    turn = TurnReply("/chat/sessions/{id}/messages")
    try:
        for ev in _answer_events(orchestrator, query, history, context):
            turn.add(ev)
    except Exception:  # the orchestrator/Azure call itself blew up
        turn.fail()
    turn.finish()
    return turn.reply()


# ---------- public API ----------

def create_session(db: Session, creator: User, title: Optional[str] = None) -> ChatSession:
    clean_title = (title or "").strip() or "New Chat"
    session = ChatSession(created_by=creator.id, title=clean_title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def list_sessions(db: Session) -> List[Dict]:
    """Three queries however many sessions there are. It used to run one User query per
    session and call len(s.messages), which loaded every message body of every session
    just to count them; sessions are shared by all users, so that grew without limit."""
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    if not sessions:
        return []
    counts = dict(db.query(ChatMessage.session_id, func.count(ChatMessage.id))
                  .group_by(ChatMessage.session_id).all())
    creator_ids = {s.created_by for s in sessions}
    creators = {u.id: u for u in db.query(User).filter(User.id.in_(creator_ids)).all()}
    out = []
    for s in sessions:
        out.append({
            "id": s.id,
            "title": s.title,
            "created_at": _iso(s.created_at),
            "updated_at": _iso(s.updated_at),
            "created_by": _user_public(creators.get(s.created_by)),
            "message_count": int(counts.get(s.id, 0)),
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



def _save_reply(db: Session, sid: int, reply: Dict) -> Optional[ChatMessage]:
    """Store the assistant's reply and bump the session's updated_at, or return None when the
    session was deleted while the answer was being written. Sessions are shared and any user
    can delete one, and the AI turn takes 10-60 s: touching the deleted ChatSession through
    the ORM raised ObjectDeletedError and the request ended in a 500 with the answer lost.
    The existence check and a Core UPDATE (0 rows, no error) avoid that."""
    if db.query(ChatSession.id).filter(ChatSession.id == sid).first() is None:
        db.rollback()
        return None
    msg = ChatMessage(session_id=sid, role="assistant", content=json.dumps(reply))
    db.add(msg)
    db.execute(_sa_update(ChatSession).where(ChatSession.id == sid)
               .values(updated_at=datetime.now(timezone.utc)))
    db.commit()
    db.refresh(msg)
    return msg

_GONE = "This chat was deleted while the answer was being written, so the answer was not saved."

_STREAM_ROUTE = "/chat/sessions/{id}/messages/stream"


def stream_message_events(db: Session, session_id: int, query: str, mode: str = "fast",
                          context: Optional[Dict[str, Any]] = None):
    """Generator of SSE-shaped event dicts for a session-backed turn — the live-progress
    counterpart to post_message(). Reuses orchestrator.answer() exactly as post_message
    does (same history, same audit gate), but yields step/sql/answer/chart/table/done AS
    THEY HAPPEN instead of consolidating them into one blocking response, then persists
    the same consolidated shape post_message would have stored, once, after "done".

    Deliberately ADDITIVE: POST /chat/sessions/{id}/messages (post_message, below) keeps
    its blocking shape and its tests. This is a new route the frontend opts into; the old
    one stays a working, tested fallback.

    The user's message is persisted FIRST, before any streaming starts — so if the
    client disconnects mid-answer (closed tab, network drop), the question they asked
    is never lost, only the reply. post_message commits it first the same way; this
    route also accumulates exactly what _run_ai_turn accumulates and writes the reply
    once the stream concludes.

    `context` is the page context after app/ai/page_context.normalise, a plain dict of
    known keys and values. It shapes this one turn only and is never stored: the next
    turn on the same shared session may come from a different page, or a different user.
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

    history = _history_for_orchestrator(db, session)

    sid = session.id   # read before the commit expires the session object
    user_msg = ChatMessage(session_id=sid, role="user", content=query)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    yield {"type": "user_message", "message": _serialize_message(user_msg)}

    turn = TurnReply(_STREAM_ROUTE)
    try:
        for ev in _answer_events(engine, query, history, context):
            # Forward every event live — "step"/"done" included, so the UI can show real
            # progress instead of a single frozen spinner label for the whole turn.
            yield turn.add(ev)
    except Exception:  # the orchestrator/Azure call itself blew up mid-stream
        yield turn.fail()
    closing = turn.finish()
    if closing:
        yield closing
    reply = turn.reply()

    assistant_msg = _save_reply(db, sid, reply)
    if assistant_msg is None:
        # Like any error after the answer, this one is partial then: the answer on screen
        # stays, with the line saying it was not saved under it.
        yield {"type": "error", "text": _GONE, **({"partial": True} if turn.answered else {})}
        return

    yield {"type": "persisted", "assistant_message": _serialize_message(assistant_msg)}


def post_message(db: Session, session_id: int, query: str,
                 context: Optional[Dict[str, Any]] = None) -> Dict:
    session = get_session_or_404(db, session_id)
    query = (query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Message text is required")

    history = _history_for_orchestrator(db, session)

    # The question is committed BEFORE the AI turn, the reply in its own commit after it.
    # One transaction around both (the user message only flushed) held SQLite's write lock
    # for the whole 10-60 s turn, so every other write in the app (sign-in, another user's
    # chat, a rename) queued behind it or failed with "database is locked". Committed
    # first, the question also survives a turn that never finishes, as it already did on
    # the streaming route. The id is read before the commit expires the session object.
    sid = session.id
    user_msg = ChatMessage(session_id=sid, role="user", content=query)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    user_out = _serialize_message(user_msg)

    # The page context is used for this turn and not stored (see stream_message_events).
    reply = _run_ai_turn(query, history, context)

    assistant_msg = _save_reply(db, sid, reply)
    if assistant_msg is None:
        raise HTTPException(status_code=404, detail=_GONE)

    return {
        "user_message": user_out,
        "assistant_message": _serialize_message(assistant_msg),
    }
