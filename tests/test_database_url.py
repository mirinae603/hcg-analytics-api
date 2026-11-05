# tests/test_database_url.py — where users and AI chat sessions are kept.
#
# Production runs in an Azure Container App, whose disk is thrown away on every deploy,
# restart and scale event. With the default SQLite file there, every AI Analyst chat and
# every signed-up user disappeared each time. These tests pin the two guards: the engine
# options a server database needs, and the warning logged when the database would sit on
# that throwaway disk.
from __future__ import annotations

from app.core import database as D

PG = "postgresql+psycopg://user:secret@db.example.com:5432/hcg?sslmode=require"
LITE = "sqlite:////app/app/data/app.db"


def test_sqlite_keeps_its_thread_setting_and_postgres_pings_idle_connections():
    assert D.engine_options(LITE) == {"connect_args": {"check_same_thread": False}}
    assert D.engine_options(PG) == {"pool_pre_ping": True}


def test_a_postgres_url_builds_an_engine_without_connecting():
    # create_engine is lazy: this proves the driver is installed and the options are valid,
    # without needing a server.
    from sqlalchemy import create_engine
    engine = create_engine(PG, **D.engine_options(PG))
    assert engine.dialect.name == "postgresql" and engine.pool._pre_ping is True
    engine.dispose()


def test_sqlite_in_a_container_app_is_warned_about():
    msg = D.ephemeral_storage_warning(LITE, {"CONTAINER_APP_NAME": "hcg-backend"})
    assert msg and "lost" in msg and "postgresql+psycopg://" in msg
    assert D.ephemeral_storage_warning(LITE, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"})


def test_no_warning_for_a_server_database_or_a_laptop():
    assert D.ephemeral_storage_warning(PG, {"CONTAINER_APP_NAME": "hcg-backend"}) is None
    assert D.ephemeral_storage_warning(LITE, {}) is None
    assert D.ephemeral_storage_warning(LITE, {"CONTAINER_APP_NAME": ""}) is None
