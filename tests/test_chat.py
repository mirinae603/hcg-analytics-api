# tests/test_chat.py — AI Analyst chat session persistence.
#
# orchestrator.answer (the real Azure-backed generator app/ai/orchestrator.py exposes,
# also used by POST /ai/chat) is monkeypatched to a deterministic fake for this whole
# file — these are unit/integration tests of the PERSISTENCE wrapper, not of Azure
# OpenAI itself, so they must run fast, free, and offline. The real function is
# exercised for real in the live smoke test (outside pytest) instead.
from __future__ import annotations

import json

import pytest

from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, bearer, signup_and_approve


@pytest.fixture(autouse=True)
def fake_orchestrator(monkeypatch):
    # "verified" mirrors the REAL orchestrator's shape exactly: "ok" | "corrected" |
    # "flagged" | None — never a plain bool. A prior version of this fixture used
    # `"verified": True`, which let chat_service.py's `bool(ev.get("verified", False))`
    # bug slip through unnoticed (bool(True) is trivially True either way) — see
    # test_flagged_answer_is_never_shown_as_verified below for the regression this was
    # actually hiding.
    def _fake_answer(query, history=None):
        yield {"type": "step", "text": "Thinking it through"}
        yield {
            "type": "answer",
            "text": f"Fake answer to: {query}",
            "verified": "ok",
            "options": ["A relevant follow-up?"],
        }
        yield {"type": "done"}

    monkeypatch.setattr("app.ai.orchestrator.answer", _fake_answer)


def _signin(client, email, password):
    r = client.post("/signin", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ---------- creation / creator tagging ----------

def test_create_chat_session_tags_creator(client, admin_headers):
    email = signup_and_approve(client, admin_headers, email="creator@example.com", password="password123")
    token = _signin(client, email, "password123")

    r = client.post("/chat/sessions", json={"title": "My first analysis"}, headers=bearer(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["title"] == "My first analysis"
    assert body["created_by"]["email"] == email
    assert body["messages"] == []


def test_create_chat_session_requires_auth(client):
    r = client.post("/chat/sessions", json={"title": "No auth"})
    assert r.status_code == 401


def test_create_chat_session_default_title(client, admin_headers):
    r = client.post("/chat/sessions", json={}, headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json()["title"]  # some non-empty default, e.g. "New Chat"


# ---------- posting messages ----------

def test_post_message_persists_user_message_and_ai_reply(client, admin_headers):
    session = client.post("/chat/sessions", json={"title": "Inventory Qs"}, headers=admin_headers).json()

    r = client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"query": "What is our current inventory turnover?"},
        headers=admin_headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["user_message"]["role"] == "user"
    assert body["user_message"]["content"] == "What is our current inventory turnover?"

    assert body["assistant_message"]["role"] == "assistant"
    assert "Fake answer to: What is our current inventory turnover?" in body["assistant_message"]["content"]["text"]
    assert body["assistant_message"]["content"]["verified"] == "ok"

    # Confirm it was actually PERSISTED, not just returned once.
    detail = client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]


def test_flagged_answer_is_never_shown_as_verified(client, admin_headers, monkeypatch):
    # Regression test for the exact bug the cross-check found: chat_service.py used to do
    # `verified = bool(ev.get("verified", False))`, and since any non-empty Python string
    # is truthy, `bool("flagged")` is True — so an answer the auditor explicitly flagged
    # as unreliable would persist and later render with a green "Verified" badge, the
    # opposite of what the badge means. It must come through as the literal string
    # "flagged", never True/truthy-as-"ok".
    def _flagged_answer(query, history=None):
        yield {"type": "answer", "text": "Uncertain figure", "verified": "flagged", "options": []}
        yield {"type": "done"}

    monkeypatch.setattr("app.ai.orchestrator.answer", _flagged_answer)
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "risky q"}, headers=admin_headers)
    assert r.status_code == 200, r.text
    verified = r.json()["assistant_message"]["content"]["verified"]
    assert verified == "flagged"
    assert verified is not True
    assert bool(verified) is not False  # sanity: "flagged" is truthy as a raw Python value...
    assert verified != "ok" and verified != "corrected"  # ...but must never be treated as a pass


def test_unverifiable_answer_has_no_verified_flag(client, admin_headers, monkeypatch):
    def _no_verify_answer(query, history=None):
        yield {"type": "answer", "text": "Can't confirm", "verified": None, "options": []}
        yield {"type": "done"}

    monkeypatch.setattr("app.ai.orchestrator.answer", _no_verify_answer)
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "q"}, headers=admin_headers)
    assert r.json()["assistant_message"]["content"]["verified"] is None


def test_table_note_and_sql_queries_are_persisted(client, admin_headers, monkeypatch):
    # Regression test for two more findings from the cross-check: the table's "note"
    # footnote and the "N queries run" transparency disclosure were both silently
    # dropped by the persistence wrapper even though the live SSE path always sent them.
    def _answer_with_table_and_sql(query, history=None):
        yield {"type": "sql", "purpose": "look up stock value", "sql": "SELECT 1", "rows": 5, "error": None}
        yield {"type": "answer", "text": "Here you go", "verified": "ok", "options": []}
        yield {"type": "table", "table": {"title": "Stock", "columns": [], "rows": []}, "note": "Showing top 50 of 120 rows."}
        yield {"type": "done"}

    monkeypatch.setattr("app.ai.orchestrator.answer", _answer_with_table_and_sql)
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "q"}, headers=admin_headers)
    content = r.json()["assistant_message"]["content"]
    assert content["table"]["note"] == "Showing top 50 of 120 rows."
    assert content["queries"] == [{"purpose": "look up stock value", "sql": "SELECT 1", "rows": 5, "error": None}]


def test_post_message_requires_auth(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "hi"})
    assert r.status_code == 401


def test_post_empty_message_rejected(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "   "}, headers=admin_headers)
    assert r.status_code in (400, 422)


def test_post_message_to_nonexistent_session_is_404(client, admin_headers):
    r = client.post("/chat/sessions/999999/messages", json={"query": "hi"}, headers=admin_headers)
    assert r.status_code == 404


# ---------- shared / common visibility across users ----------

def test_different_logged_in_user_can_list_and_see_who_created_session(client, admin_headers):
    creator_email = signup_and_approve(client, admin_headers, email="chat-creator@example.com", password="password123")
    creator_token = _signin(client, creator_email, "password123")

    session = client.post("/chat/sessions", json={"title": "Creator's session"}, headers=bearer(creator_token)).json()
    client.post(
        f"/chat/sessions/{session['id']}/messages",
        json={"query": "Any wastage risk this month?"},
        headers=bearer(creator_token),
    )

    other_email = signup_and_approve(client, admin_headers, email="chat-viewer@example.com", password="password123")
    other_token = _signin(client, other_email, "password123")

    # A DIFFERENT user (not the creator, not even an admin) can list it...
    listing = client.get("/chat/sessions", headers=bearer(other_token))
    assert listing.status_code == 200, listing.text
    ids = {s["id"]: s for s in listing.json()["sessions"]}
    assert session["id"] in ids
    assert ids[session["id"]]["created_by"]["email"] == creator_email

    # ...and open it, seeing the full message history and the creator's identity.
    detail = client.get(f"/chat/sessions/{session['id']}", headers=bearer(other_token))
    assert detail.status_code == 200, detail.text
    detail_body = detail.json()
    assert detail_body["created_by"]["email"] == creator_email
    assert any(m["role"] == "user" and "wastage" in m["content"].lower() for m in detail_body["messages"])


def test_admin_can_see_a_members_session(client, admin_headers):
    member_email = signup_and_approve(client, admin_headers, email="member-chat@example.com", password="password123")
    member_token = _signin(client, member_email, "password123")

    session = client.post("/chat/sessions", json={"title": "Member's own session"}, headers=bearer(member_token)).json()

    r = client.get("/chat/sessions", headers=admin_headers)
    assert r.status_code == 200
    found = [s for s in r.json()["sessions"] if s["id"] == session["id"]]
    assert found and found[0]["created_by"]["email"] == member_email


# ---------- listing / not-found edge cases ----------

def test_empty_chat_session_list_handled_gracefully(client, admin_headers):
    r = client.get("/chat/sessions", headers=admin_headers)
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


def test_list_chat_sessions_requires_auth(client):
    r = client.get("/chat/sessions")
    assert r.status_code == 401


def test_get_nonexistent_session_is_404(client, admin_headers):
    r = client.get("/chat/sessions/424242", headers=admin_headers)
    assert r.status_code == 404


# ---------- delete ----------

def test_delete_session_removes_it_and_its_messages(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    client.post(f"/chat/sessions/{session['id']}/messages", json={"query": "hi"}, headers=admin_headers)

    r = client.delete(f"/chat/sessions/{session['id']}", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": True, "id": session["id"]}

    assert client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).status_code == 404
    ids = {s["id"] for s in client.get("/chat/sessions", headers=admin_headers).json()["sessions"]}
    assert session["id"] not in ids


def test_delete_nonexistent_session_is_404(client, admin_headers):
    r = client.delete("/chat/sessions/424242", headers=admin_headers)
    assert r.status_code == 404


def test_delete_session_requires_auth(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.delete(f"/chat/sessions/{session['id']}")
    assert r.status_code == 401


def test_any_user_can_delete_any_session(client, admin_headers):
    # "Chat is common": deletion follows the same shared-visibility model as list/view,
    # already covered for those by test_different_logged_in_user_can_list_and_see_who_created_session.
    creator_email = signup_and_approve(client, admin_headers, email="del-creator@example.com", password="password123")
    creator_token = _signin(client, creator_email, "password123")
    session = client.post("/chat/sessions", json={}, headers=bearer(creator_token)).json()

    other_email = signup_and_approve(client, admin_headers, email="del-other@example.com", password="password123")
    other_token = _signin(client, other_email, "password123")

    r = client.delete(f"/chat/sessions/{session['id']}", headers=bearer(other_token))
    assert r.status_code == 200, r.text
    assert client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).status_code == 404


# ---------- rename ----------

def test_rename_session(client, admin_headers):
    session = client.post("/chat/sessions", json={"title": "Original"}, headers=admin_headers).json()
    r = client.patch(f"/chat/sessions/{session['id']}", json={"title": "Renamed"}, headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"id": session["id"], "title": "Renamed"}
    assert client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).json()["title"] == "Renamed"


def test_rename_does_not_reorder_updated_at(client, admin_headers):
    # A rename must not bump updated_at -- that would jump the session to "Today" in the
    # sidebar purely from a title edit, with no new conversation activity behind it.
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    before = client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).json()["updated_at"]
    client.patch(f"/chat/sessions/{session['id']}", json={"title": "New name"}, headers=admin_headers)
    after = client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).json()["updated_at"]
    assert before == after


def test_rename_blank_title_rejected(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.patch(f"/chat/sessions/{session['id']}", json={"title": "   "}, headers=admin_headers)
    assert r.status_code in (400, 422)


def test_rename_nonexistent_session_is_404(client, admin_headers):
    r = client.patch("/chat/sessions/424242", json={"title": "x"}, headers=admin_headers)
    assert r.status_code == 404


def test_rename_requires_auth(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.patch(f"/chat/sessions/{session['id']}", json={"title": "x"})
    assert r.status_code == 401


# ---------- streaming endpoint ----------
# Additive alongside POST .../messages (never modified): live step/sql/answer/done
# events instead of one blocking response, persisting the same shape at the end.

def _read_sse_events(response):
    events = []
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == '{"type": "end"}':
            continue
        events.append(json.loads(payload))
    return events


def test_stream_message_yields_live_events_and_persists(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    with client.stream(
        "POST", f"/chat/sessions/{session['id']}/messages/stream",
        json={"query": "What is our current inventory turnover?"}, headers=admin_headers,
    ) as r:
        assert r.status_code == 200
        events = _read_sse_events(r)

    types = [e["type"] for e in events]
    assert "user_message" in types
    assert "step" in types  # the whole point: live progress, not a silent wait
    assert "answer" in types
    assert "persisted" in types

    persisted = next(e for e in events if e["type"] == "persisted")
    assert persisted["assistant_message"]["role"] == "assistant"
    assert "Fake answer to" in persisted["assistant_message"]["content"]["text"]

    detail = client.get(f"/chat/sessions/{session['id']}", headers=admin_headers).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


def test_stream_message_to_nonexistent_session_is_404_not_a_broken_stream(client, admin_headers):
    r = client.post(
        "/chat/sessions/424242/messages/stream",
        json={"query": "hi"}, headers=admin_headers,
    )
    assert r.status_code == 404


def test_stream_empty_message_is_400(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(
        f"/chat/sessions/{session['id']}/messages/stream",
        json={"query": "   "}, headers=admin_headers,
    )
    assert r.status_code in (400, 422)


def test_stream_message_requires_auth(client, admin_headers):
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()
    r = client.post(f"/chat/sessions/{session['id']}/messages/stream", json={"query": "hi"})
    assert r.status_code == 401


def test_stream_scope_threads_into_next_turn(client, admin_headers, monkeypatch):
    # Regression test for the exact live bug this was built to fix: a follow-up used to
    # get ZERO prior context because chat_service silently dropped the orchestrator's
    # `scope` field instead of persisting it, so _latest_scope(history) always saw "".
    seen_scope_in_history = {}

    def _scoped_answer(query, history=None):
        hist_text = json.dumps(history or [])
        seen_scope_in_history[query] = "active scope" in hist_text
        yield {"type": "answer", "text": f"reply to {query}", "verified": "ok",
               "options": [], "scope": "category = INJECTIONS"}
        yield {"type": "done"}

    monkeypatch.setattr("app.ai.orchestrator.answer", _scoped_answer)
    session = client.post("/chat/sessions", json={}, headers=admin_headers).json()

    with client.stream("POST", f"/chat/sessions/{session['id']}/messages/stream",
                        json={"query": "injection turnover"}, headers=admin_headers) as r:
        _read_sse_events(r)
    with client.stream("POST", f"/chat/sessions/{session['id']}/messages/stream",
                        json={"query": "and last month?"}, headers=admin_headers) as r:
        _read_sse_events(r)

    assert seen_scope_in_history["injection turnover"] is False  # nothing to inherit yet
    assert seen_scope_in_history["and last month?"] is True  # must see turn 1's scope marker
