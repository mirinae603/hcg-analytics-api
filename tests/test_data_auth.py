# tests/test_data_auth.py — the dashboard's data needs a login.
#
# Until this change only /chat and the /admin user routes checked the bearer token. Every
# KPI, forecast, executive-summary and AI Analyst route answered anyone who had the backend
# URL (observed on the live Container App), and CORS allowed every origin. Soon the same
# API serves doctor-level formulary compliance, which must not be public.
#
# The rule is enforced once, in app/api/api_router.py, as a router dependency. The route
# inventory test below is what keeps a NEW router from quietly shipping without it.
from __future__ import annotations

import re

import pytest

from app.core.config import settings
from app.main import app

from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, bearer

# The only routes a signed-out caller may use.
OPEN_ROUTES = {("GET", "/health"), ("POST", "/signin"), ("POST", "/signup")}


@pytest.mark.parametrize("method, path", [
    ("GET", "/kpi/current-stock-value"),
    ("GET", "/meta/plants"),
    ("GET", "/api/dashboard/all"),
    ("GET", "/ai/status"),
    ("GET", "/ai/mentions"),
    ("POST", "/ai/chat"),
])
def test_data_routes_refuse_a_caller_without_a_token(anon_client, method, path):
    r = anon_client.request(method, path, json={"query": "hi"} if method == "POST" else None)
    assert r.status_code == 401, r.text


def test_data_routes_refuse_a_forged_token(anon_client):
    r = anon_client.get("/meta/plants", headers=bearer("this-is-not-a-jwt"))
    assert r.status_code == 401


def test_health_stays_open(anon_client):
    r = anon_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_signin_stays_open_and_its_token_opens_the_data(anon_client):
    r = anon_client.post("/signin", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert r.status_code == 200, r.text
    r = anon_client.get("/meta/plants", headers=bearer(r.json()["token"]))
    assert r.status_code == 200, r.text
    assert any(p["code"] == "ALL" for p in r.json()["plants"])


def test_the_default_test_client_is_signed_in(client):
    r = client.get("/meta/plants")
    assert r.status_code == 200, r.text


def test_refresh_needs_both_a_login_and_the_admin_token(client, anon_client):
    admin_token = {"x-admin-token": settings.ADMIN_REFRESH_TOKEN}
    assert anon_client.post("/admin/refresh-data", headers=admin_token).status_code == 401
    assert client.post("/admin/refresh-data", headers={"x-admin-token": "wrong"}).status_code == 401
    r = client.post("/admin/refresh-data", headers=admin_token)
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "cache cleared"}


def test_every_route_except_health_and_sign_in_needs_a_login(anon_client):
    # Walk the published schema (newer FastAPI no longer flattens included routers into
    # app.routes) and call each route signed out. The login dependency runs before the
    # handler and before parameter validation, so a placeholder path value is enough.
    unprotected = []
    for path, ops in app.openapi()["paths"].items():
        for method in ops:
            method = method.upper()
            if (method, path) in OPEN_ROUTES:
                continue
            url = re.sub(r"\{[^}]+\}", "1", path)
            r = anon_client.request(method, url, json=None if method == "GET" else {})
            if r.status_code != 401:
                unprotected.append(f"{method} {path} -> {r.status_code}")
    assert len(app.openapi()["paths"]) > 50  # the walk really saw the app's routes
    assert unprotected == [], (
        "these routes answer without a login — include their router with "
        f"dependencies=_login in app/api/api_router.py: {unprotected}")


# ---------- CORS: only the dashboard's own origins ----------

def _preflight(anon_client, origin: str):
    return anon_client.options("/meta/plants", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })


@pytest.mark.parametrize("origin", [
    "https://hcg.bidezy.com",
    "http://localhost:3000",
    "https://white-glacier-06a29bc00.azurestaticapps.net",
    "https://white-glacier-06a29bc00-12.centralindia.azurestaticapps.net",  # PR preview
])
def test_the_dashboard_origins_may_call_the_api(anon_client, origin):
    r = _preflight(anon_client, origin)
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == origin
    # bearer header, no cookies: credentialed CORS stays off
    assert "access-control-allow-credentials" not in r.headers


def test_other_origins_are_refused(anon_client):
    r = _preflight(anon_client, "https://evil.example.com")
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers
