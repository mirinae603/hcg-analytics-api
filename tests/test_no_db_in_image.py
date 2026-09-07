# tests/test_no_db_in_image.py — the application database must never ship in the image.
#
# Dockerfile does `COPY . .`, and .dockerignore did not exclude app/data/*.db. So every
# deployed image carried the LOCAL SQLite database: five throwaway test accounts
# (finalcheck@example.com, regress.tester@example.com, uidesign.audit@example.com …) and
# 24 debugging chat sessions were shipped to production, and each deploy overwrote whatever
# real users and history production had accumulated with a developer's laptop state.
#
# .gitignore already excluded these files. .dockerignore is a separate list, and nothing
# tied the two together — which is exactly the kind of gap that stays invisible until
# somebody reads a production user list and finds their own test fixtures in it.
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dockerignore() -> list[str]:
    text = (ROOT / ".dockerignore").read_text()
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_the_sqlite_database_is_excluded_from_the_image():
    patterns = _dockerignore()
    assert any(p in ("app/data/*.db", "app/data/app.db", "app/data/") for p in patterns), (
        "app/data/*.db is not in .dockerignore — `COPY . .` will bake the local users and "
        "chat history into the deployed image")


def test_the_legacy_user_store_is_excluded_too():
    assert any("users.json" in p for p in _dockerignore())


def test_the_database_is_created_at_startup_so_an_empty_image_is_safe():
    # excluding the file is only safe because the app builds and seeds its own DB
    main = (ROOT / "app" / "main.py").read_text()
    assert "seed_and_migrate" in main
    assert "create_all" in main or "create_all" in (
        ROOT / "app" / "core" / "seed.py").read_text()


def test_dockerignore_and_gitignore_agree_about_the_database():
    # they are separate lists; the whole bug was that only one of them knew
    git = (ROOT / ".gitignore").read_text()
    assert "app/data/*.db" in git or "app/data/app.db" in git
    assert any("app/data/" in p and ".db" in p for p in _dockerignore())
