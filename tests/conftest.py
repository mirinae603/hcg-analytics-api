# tests/conftest.py — shared pytest fixtures.
#
# Every test gets a FRESH, ISOLATED SQLite DB (a tmp_path file, never the real
# app/data/app.db) wired in via FastAPI's dependency_overrides, pre-seeded with
# admin@hcg.com through the exact same seeding function production startup uses
# (app.core.seed._ensure_seed_admin) — so the all-important "admin still logs in and
# is role=admin" behaviour is tested against real seeding code, not a hand-rolled
# stand-in. TestClient is used WITHOUT the `with` context manager on purpose: that
# skips FastAPI's startup event (verified empirically), so importing/exercising the
# app in tests never touches the real on-disk DB as a side effect.
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.database import Base, get_db  # noqa: E402
from app.core.seed import _ensure_seed_admin  # noqa: E402
from app.main import app  # noqa: E402

ADMIN_EMAIL = "admin@hcg.com"
ADMIN_PASSWORD = "admin123"


@pytest.fixture()
def db_session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    yield TestSessionLocal
    engine.dispose()


@pytest.fixture()
def client(db_session_factory):
    def _override_get_db():
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db

    seed_db = db_session_factory()
    try:
        _ensure_seed_admin(seed_db)
    finally:
        seed_db.close()

    test_client = TestClient(app)
    yield test_client

    app.dependency_overrides.clear()


@pytest.fixture()
def admin_token(client):
    r = client.post("/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture()
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


# ---------- shared helpers ----------

def signup(client, email="member@example.com", password="password123", first="Mem", last="Ber"):
    return client.post(
        "/signup",
        json={"firstName": first, "lastName": last, "email": email, "password": password},
    )


def approve(client, admin_headers, email):
    return client.post(
        "/admin/approve-user",
        json={"email": email, "action": "approve"},
        headers=admin_headers,
    )


def signup_and_approve(client, admin_headers, email="member@example.com", password="password123", first="Mem", last="Ber"):
    r = signup(client, email=email, password=password, first=first, last=last)
    assert r.status_code == 200, r.text
    approve(client, admin_headers, email)
    return email


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}
