# tests/test_db_portability.py — the documented escape hatch has to actually work.
#
# A Container App filesystem is EPHEMERAL. Chat written in production is lost on every
# restart, deploy or scale event, and a second replica would get its own separate SQLite
# file with its own separate history. So the store has to be movable off SQLite.
#
# app/core/config.py has always said DATABASE_URL was swappable "e.g. to point at Postgres
# later" — and with no driver installed that swap died with ModuleNotFoundError. A promise
# in a comment is not a migration path.
from __future__ import annotations

from pathlib import Path

import sqlalchemy

ROOT = Path(__file__).resolve().parents[1]


def test_a_postgres_url_can_be_constructed():
    # no connection is attempted — this proves the driver is installed and importable
    sqlalchemy.create_engine("postgresql+psycopg://user:pw@host:5432/db")


def test_the_driver_is_declared_for_the_image():
    assert "psycopg" in (ROOT / "requirements.txt").read_text()


def test_the_url_is_read_from_the_environment():
    cfg = (ROOT / "app" / "core" / "config.py").read_text()
    assert "DATABASE_URL" in cfg


def test_sqlite_specific_args_are_applied_only_to_sqlite():
    # check_same_thread is a SQLite-only connect arg; passing it to Postgres raises
    db = (ROOT / "app" / "core" / "database.py").read_text()
    assert 'startswith("sqlite")' in db
