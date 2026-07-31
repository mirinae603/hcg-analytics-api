# tests/test_auth.py — signup/signin, JWT, and admin-only protection.
#
# WRITTEN BEFORE ANY IMPLEMENTATION EXISTED (TDD) — see the session's red-run output
# for proof every test here failed the first time it was run.
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest

from app.core.config import settings
from app.core.jwt_auth import decode_access_token
from app.models import User

from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, approve, bearer, signup, signup_and_approve


# ---------- signup ----------

def test_signup_creates_pending_member(client):
    r = signup(client, email="new.member@example.com")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["email"] == "new.member@example.com"
    assert body["user"]["status"] == "pending"
    assert body["user"]["role"] == "member"


def test_signup_duplicate_email_is_rejected_with_clear_error(client):
    r1 = signup(client, email="dup@example.com")
    assert r1.status_code == 200, r1.text

    r2 = signup(client, email="dup@example.com")
    assert r2.status_code == 409
    assert "already exists" in r2.json()["detail"].lower()


def test_signup_rejects_short_password(client):
    r = signup(client, email="short@example.com", password="123")
    assert r.status_code == 400
    assert "password" in r.json()["detail"].lower()


# ---------- signin: pending / rejected / wrong credentials ----------

def test_signin_fails_for_pending_user_with_clear_message(client):
    signup(client, email="pending@example.com", password="password123")

    r = client.post("/signin", json={"email": "pending@example.com", "password": "password123"})
    assert r.status_code == 403
    detail = r.json()["detail"].lower()
    assert "pending" in detail or "approv" in detail


def test_signin_fails_for_rejected_user_with_clear_message(client, admin_headers):
    signup(client, email="rejected@example.com", password="password123")
    r = client.post("/admin/approve-user", json={"email": "rejected@example.com", "action": "reject"}, headers=admin_headers)
    assert r.status_code == 200, r.text

    r = client.post("/signin", json={"email": "rejected@example.com", "password": "password123"})
    assert r.status_code == 403
    assert "reject" in r.json()["detail"].lower()


def test_signin_wrong_password_rejected_not_500(client):
    signup(client, email="wrongpass@example.com", password="password123")
    r = client.post("/signin", json={"email": "wrongpass@example.com", "password": "totally-wrong"})
    assert r.status_code == 401
    assert "invalid" in r.json()["detail"].lower()


def test_signin_unknown_email_rejected_not_500(client):
    r = client.post("/signin", json={"email": "nobody@example.com", "password": "whatever123"})
    assert r.status_code == 401


# ---------- signin success + JWT shape ----------

def test_signin_succeeds_for_approved_user_and_returns_valid_jwt(client, admin_headers):
    email = signup_and_approve(client, admin_headers, email="approved@example.com", password="password123")

    r = client.post("/signin", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["message"]
    assert body["token_type"] == "bearer"
    token = body["token"]
    assert isinstance(token, str) and token.count(".") == 2  # header.payload.signature

    # Must actually verify against our own secret/algorithm — not just look like a JWT.
    payload = decode_access_token(token)
    assert payload["email"] == email


def test_jwt_decodes_to_correct_user_id_and_role(client, admin_headers, db_session_factory):
    email = signup_and_approve(client, admin_headers, email="claims@example.com", password="password123")

    r = client.post("/signin", json={"email": email, "password": "password123"})
    token = r.json()["token"]

    payload = decode_access_token(token)

    db = db_session_factory()
    try:
        user = db.query(User).filter(User.email == email).first()
    finally:
        db.close()

    assert int(payload["sub"]) == user.id
    assert payload["role"] == "member"
    assert payload["status"] == "approved"


def test_admin_seed_account_signs_in_and_resolves_to_admin_role(client):
    """THE single most important test in this suite: admin@hcg.com / admin123 must
    keep working after the whole auth rework, and must resolve to role=admin."""
    r = client.post("/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["email"] == ADMIN_EMAIL
    assert body["user"]["role"] == "admin"
    assert body["user"]["status"] == "approved"

    payload = decode_access_token(body["token"])
    assert payload["role"] == "admin"
    assert payload["email"] == ADMIN_EMAIL


# ---------- admin-only endpoint protection ----------

def test_member_gets_403_on_admin_pending_users(client, admin_headers):
    email = signup_and_approve(client, admin_headers, email="member1@example.com", password="password123")
    r = client.post("/signin", json={"email": email, "password": "password123"})
    member_token = r.json()["token"]

    r = client.get("/admin/pending-users", headers=bearer(member_token))
    assert r.status_code == 403


def test_member_gets_403_on_admin_users_list(client, admin_headers):
    email = signup_and_approve(client, admin_headers, email="member2@example.com", password="password123")
    member_token = client.post("/signin", json={"email": email, "password": "password123"}).json()["token"]

    r = client.get("/admin/users", headers=bearer(member_token))
    assert r.status_code == 403


def test_member_gets_403_on_admin_approve_user(client, admin_headers):
    email = signup_and_approve(client, admin_headers, email="member3@example.com", password="password123")
    member_token = client.post("/signin", json={"email": email, "password": "password123"}).json()["token"]
    signup(client, email="someone-else@example.com", password="password123")

    r = client.post(
        "/admin/approve-user",
        json={"email": "someone-else@example.com", "action": "approve"},
        headers=bearer(member_token),
    )
    assert r.status_code == 403


def test_admin_can_list_pending_users(client, admin_headers):
    signup(client, email="pending-list@example.com", password="password123")

    r = client.get("/admin/pending-users", headers=admin_headers)
    assert r.status_code == 200, r.text
    emails = [u["email"] for u in r.json()["users"]]
    assert "pending-list@example.com" in emails


def test_admin_can_list_all_users(client, admin_headers):
    r = client.get("/admin/users", headers=admin_headers)
    assert r.status_code == 200, r.text
    emails = [u["email"] for u in r.json()["users"]]
    assert ADMIN_EMAIL in emails


def test_admin_approve_flips_status_and_user_can_then_signin(client, admin_headers):
    signup(client, email="toapprove@example.com", password="password123")

    # not yet approved -> signin blocked
    r = client.post("/signin", json={"email": "toapprove@example.com", "password": "password123"})
    assert r.status_code == 403

    r = approve(client, admin_headers, "toapprove@example.com")
    assert r.status_code == 200, r.text
    assert r.json()["user"]["status"] == "approved"

    r = client.post("/signin", json={"email": "toapprove@example.com", "password": "password123"})
    assert r.status_code == 200, r.text
    assert r.json()["user"]["status"] == "approved"


# ---------- token edge cases ----------

def test_admin_endpoint_without_token_is_401(client):
    r = client.get("/admin/users")
    assert r.status_code == 401


def test_admin_endpoint_with_malformed_token_is_401(client):
    r = client.get("/admin/users", headers=bearer("this-is-not-a-jwt"))
    assert r.status_code == 401


def test_admin_endpoint_with_wrong_signature_token_is_401(client):
    bad_token = pyjwt.encode(
        {"sub": "1", "email": ADMIN_EMAIL, "role": "admin", "status": "approved",
         "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        "a-completely-different-secret",
        algorithm=settings.JWT_ALGORITHM,
    )
    r = client.get("/admin/users", headers=bearer(bad_token))
    assert r.status_code == 401


def test_admin_endpoint_with_expired_token_is_401(client, admin_headers, db_session_factory):
    db = db_session_factory()
    try:
        admin = db.query(User).filter(User.email == ADMIN_EMAIL).first()
        admin_id = admin.id
    finally:
        db.close()

    expired_token = pyjwt.encode(
        {
            "sub": str(admin_id),
            "email": ADMIN_EMAIL,
            "role": "admin",
            "status": "approved",
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    r = client.get("/admin/users", headers=bearer(expired_token))
    assert r.status_code == 401
    assert "expired" in r.json()["detail"].lower()


def test_me_endpoint_returns_current_user(client, admin_headers):
    r = client.get("/me", headers=admin_headers)
    assert r.status_code == 200, r.text
    assert r.json()["user"]["email"] == ADMIN_EMAIL
    assert r.json()["user"]["role"] == "admin"


# ---------- legacy sha256 fallback (defensive — see app/core/security.py) ----------

def test_legacy_sha256_hash_verifies_once_then_upgrades_to_bcrypt(client, db_session_factory):
    legacy_password = "legacyPass123"
    legacy_hash = hashlib.sha256(legacy_password.encode()).hexdigest()

    db = db_session_factory()
    try:
        db.add(User(
            first_name="Legacy", last_name="User", email="legacy@example.com",
            password_hash=legacy_hash, role="member", status="approved",
        ))
        db.commit()
    finally:
        db.close()

    r = client.post("/signin", json={"email": "legacy@example.com", "password": legacy_password})
    assert r.status_code == 200, r.text

    db = db_session_factory()
    try:
        user = db.query(User).filter(User.email == "legacy@example.com").first()
        assert user.password_hash != legacy_hash  # upgraded
        assert user.password_hash.startswith("$2b$") or user.password_hash.startswith("$2a$")
    finally:
        db.close()

    # Second sign-in now goes through the normal bcrypt path and still works.
    r2 = client.post("/signin", json={"email": "legacy@example.com", "password": legacy_password})
    assert r2.status_code == 200, r2.text
