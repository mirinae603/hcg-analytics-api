# app/core/seed.py — one-time DB seed + migration from the old app/data/users.json.
#
# Runs at app startup (see app/main.py's startup hook). Idempotent — safe to call on
# every boot.
#
#   1. Ensures admin@hcg.com exists with role=admin, status=approved. If already
#      present it's re-affirmed (role/status corrected if somehow drifted) but its
#      password_hash is left alone once it exists — only created fresh the first time.
#   2. Migrates any OTHER rows already sitting in app/data/users.json (there are none
#      as of writing — the file was checked and contains only the seed admin — but this
#      keeps existing accounts, if any ever exist, from being silently dropped instead
#      of trusting that check to stay true forever).
#      Their existing password_hash (old unsalted sha256) is preserved as-is; sign-in's
#      legacy fallback (app/core/security.verify_password_with_legacy_fallback) verifies
#      it once and transparently upgrades it to bcrypt on that successful login.
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.database import SessionLocal, create_all
from app.core.security import hash_password
from app.models import User, UserRole, UserStatus

SEED_ADMIN_EMAIL = "admin@hcg.com"
SEED_ADMIN_PASSWORD = "admin123"  # plaintext known — re-hashed fresh with bcrypt, never migrated from the old sha256

USERS_JSON_PATH = Path(__file__).resolve().parents[1] / "data" / "users.json"


def _ensure_seed_admin(db: Session) -> None:
    admin = db.query(User).filter(User.email == SEED_ADMIN_EMAIL).first()
    if admin is None:
        admin = User(
            first_name="HCG",
            last_name="Admin",
            email=SEED_ADMIN_EMAIL,
            password_hash=hash_password(SEED_ADMIN_PASSWORD),
            role=UserRole.ADMIN,
            status=UserStatus.APPROVED,
        )
        db.add(admin)
        db.commit()
        logging.info("Seeded default admin user (%s) with a fresh bcrypt hash.", SEED_ADMIN_EMAIL)
        return

    # Already exists (e.g. re-boot) — make sure it never drifts off admin/approved.
    changed = False
    if admin.role != UserRole.ADMIN:
        admin.role = UserRole.ADMIN
        changed = True
    if admin.status != UserStatus.APPROVED:
        admin.status = UserStatus.APPROVED
        changed = True
    if changed:
        db.commit()


def _migrate_legacy_users_json(db: Session) -> int:
    """Import any users.json rows not yet in the DB. Returns count migrated."""
    if not USERS_JSON_PATH.exists():
        return 0
    try:
        with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
            legacy_users = json.load(f)
    except Exception as e:
        logging.error("Could not read legacy users.json for migration: %s", e)
        return 0

    migrated = 0
    for u in legacy_users:
        email = str(u.get("email", "")).strip()
        if not email or email.lower() == SEED_ADMIN_EMAIL:
            continue  # admin is handled by _ensure_seed_admin with a fresh hash
        if db.query(User).filter(User.email == email).first() is not None:
            continue
        legacy_hash = u.get("password_hash", "")
        if not legacy_hash:
            continue  # nothing usable to migrate
        db.add(
            User(
                first_name=u.get("firstName", "") or "",
                last_name=u.get("lastName", "") or "",
                email=email,
                password_hash=legacy_hash,  # old sha256 — verified via legacy fallback, upgraded on next login
                role=UserRole.MEMBER,
                status=u.get("status", UserStatus.PENDING),
            )
        )
        migrated += 1
    if migrated:
        db.commit()
        logging.info("Migrated %d legacy user(s) from users.json into the DB.", migrated)
    return migrated


def seed_and_migrate() -> None:
    create_all()
    db = SessionLocal()
    try:
        _ensure_seed_admin(db)
        _migrate_legacy_users_json(db)
    finally:
        db.close()
