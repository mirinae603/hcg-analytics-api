# tests/test_cache_warmup.py — the background cache warm-up (app/api/warmup.py) after a start and
# after POST /admin/refresh-data.
#
# The work itself is replaced by a stub in every test but the last, which runs it on the
# executive overview's own fixture (tests/test_executive_api.py) to show it fills the very cache
# entry a visitor's first request reads. The module's state is reset around every test, and any
# thread a test starts is joined before the next one.
from __future__ import annotations

import logging
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import dashboard_summary as ds
from app.api import executive as E
from app.api import warmup as W
from app.core.config import settings

from tests.test_executive_api import AUG, data  # noqa: F401  (data: the executive overview's fixture)

THREAD = "cache-warmup"


def _join() -> None:
    for t in threading.enumerate():
        if t.name == THREAD:
            t.join(10)
            assert not t.is_alive(), "the warm-up thread did not finish"


@pytest.fixture()
def calls(monkeypatch):
    """A fresh process (not serving), plenty of memory, and a stub for the work that records the
    thread it ran on."""
    monkeypatch.setattr(W, "_serving", False)
    monkeypatch.setattr(W, "_running", False)
    monkeypatch.setattr(W, "_again", False)
    monkeypatch.setattr(W, "free_mb", lambda: 10_000.0)
    done: list = []
    monkeypatch.setattr(W, "warm", lambda: done.append(threading.current_thread().name))
    yield done
    _join()


def _refresh(monkeypatch) -> dict:
    monkeypatch.setattr(ds.da, "refresh_cache", lambda: None)     # the data caches are not the point here
    return ds.refresh_data(x_admin_token=settings.ADMIN_REFRESH_TOKEN)


# ---------- when it runs ----------

def test_nothing_is_warmed_outside_a_serving_process(calls, monkeypatch):
    # what every other test sees: the app is built without its lifespan, so a refresh warms nothing
    assert W.start("startup") is False
    assert _refresh(monkeypatch) == {"status": "cache cleared"}
    _join()
    assert calls == [] and not W._running
    assert THREAD not in [t.name for t in threading.enumerate()]


def test_a_serving_process_warms_in_the_background(calls):
    W.serving()
    assert W.start("startup") is True
    _join()
    assert calls == [THREAD]                       # on its own thread, never the caller's
    assert not W._running and not W._again


def test_a_refresh_starts_a_warm_up_and_answers_as_before(calls, monkeypatch):
    W.serving()
    assert _refresh(monkeypatch) == {"status": "cache cleared"}
    _join()
    assert calls == [THREAD]


def test_the_lifespan_marks_the_server_and_starts_a_warm_up(monkeypatch):
    from app.core import seed
    from app.main import app
    seen: list = []
    monkeypatch.setattr(seed, "seed_and_migrate", lambda: None)  # the real app.db is never touched
    monkeypatch.setattr(W, "serving", lambda: seen.append("serving"))
    monkeypatch.setattr(W, "start", lambda reason: seen.append(reason) or True)
    with TestClient(app):
        pass
    assert seen == ["serving", "startup"]


def test_asked_again_while_running_it_runs_once_more(calls, monkeypatch):
    # a refresh while the start-up warm-up is at work: once more when it is done, not once per ask
    gate, first = threading.Event(), threading.Event()

    def slow():
        calls.append(threading.current_thread().name)
        first.set()
        assert gate.wait(10)

    monkeypatch.setattr(W, "warm", slow)
    W.serving()
    assert W.start("startup") is True
    assert first.wait(10)
    assert W.start("refresh") is False and W.start("refresh") is False
    gate.set()
    _join()
    assert calls == [THREAD, THREAD] and not W._running and not W._again


# ---------- it never fails a start or a request ----------

def test_a_failure_is_logged_and_the_next_start_works(calls, monkeypatch, caplog):
    def boom():
        raise RuntimeError("table unreadable")

    monkeypatch.setattr(W, "warm", boom)
    W.serving()
    with caplog.at_level(logging.INFO, logger=W.__name__):
        assert W.start("startup") is True
        _join()
        assert [r.levelname for r in caplog.records] == ["WARNING"]
        assert "cache warm-up (startup) failed" in caplog.records[0].getMessage()
        # a section not built (503) is no failure: said in one line, no traceback
        caplog.clear()
        monkeypatch.setattr(W, "warm", lambda: (_ for _ in ()).throw(HTTPException(503, "not built")))
        assert W.start("refresh") is True
        _join()
        assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
            ("INFO", "cache warm-up (refresh) skipped: not built")]
    assert not W._running


def test_too_little_memory_warms_nothing(calls, monkeypatch, caplog):
    monkeypatch.setattr(W, "free_mb", lambda: W.WARM_NEEDS_MB - 1.0)
    W.serving()
    with caplog.at_level(logging.INFO, logger=W.__name__):
        assert W.start("startup") is True
        _join()
    assert calls == []
    assert "skipped: 599 MB free, 1450 MB needed (600 to warm, 850 kept for the AI Analyst)" in caplog.text
    # memory that cannot be read does not stop it (a developer's machine without /proc)
    monkeypatch.setattr(W, "free_mb", lambda: None)
    assert W.start("startup") is True
    _join()
    assert calls == [THREAD]


def test_room_to_warm_but_not_for_the_ai_after_it_warms_nothing(calls, monkeypatch, caplog):
    # 13-Nov-2026: a container with room for the warm-up alone was warmed, and the AI Analyst's
    # first answer on top of it got the server killed. Warming now needs room for both.
    W.serving()
    for free in (W.WARM_NEEDS_MB, W.WARM_NEEDS_MB + W.AI_RESERVE_MB - 1.0):
        monkeypatch.setattr(W, "free_mb", lambda free=free: free)
        with caplog.at_level(logging.INFO, logger=W.__name__):
            assert W.start("startup") is True
            _join()
        assert calls == []
    assert "kept for the AI Analyst" in caplog.text
    monkeypatch.setattr(W, "free_mb", lambda: float(W.WARM_NEEDS_MB + W.AI_RESERVE_MB))
    assert W.start("startup") is True
    _join()
    assert calls == [THREAD]


# ---------- how much memory is free ----------

def _files(tmp_path, monkeypatch, v2=None, v1=None, meminfo=None):
    """The container's cgroup files (v2 or v1, as (limit, usage)) and /proc/meminfo, in tmp_path."""
    paths = []
    for name, pair in (("v2", v2), ("v1", v1)):
        limit, usage = tmp_path / f"{name}.limit", tmp_path / f"{name}.usage"
        if pair is not None:
            limit.write_text(f"{pair[0]}\n")
            usage.write_text(f"{pair[1]}\n")
        paths.append((str(limit), str(usage)))
    monkeypatch.setattr(W, "CGROUP_FILES", tuple(paths))
    mem = tmp_path / "meminfo"
    if meminfo is not None:
        mem.write_text(f"MemTotal:       8000000 kB\nMemAvailable:   {meminfo} kB\n")
    monkeypatch.setattr(W, "MEMINFO", str(mem))


GIB, MIB = 2 ** 30, 2 ** 20


@pytest.mark.parametrize("v2, v1, meminfo, expected", [
    ((GIB, 512 * MIB), None, 4_096_000, 512.0),             # a 1 GiB container using 512 MiB
    (("max", 512 * MIB), None, 2_048_000, 2000.0),          # no limit set: the machine decides
    (None, (2 * GIB, 1536 * MIB), 4_096_000, 512.0),        # cgroup v1
    (None, (9223372036854771712, MIB), 1_024_000, 1000.0),  # v1's 'no limit'
    ((512 * MIB, 600 * MIB), None, 4_096_000, -88.0),       # over its limit already: nothing free
    (("garbage", "x"), None, 1_024_000, 1000.0),            # unreadable: the next source
    (None, None, 512_000, 500.0),
    (None, None, None, None),                               # nothing to read: unknown
])
def test_free_memory_is_the_containers_then_the_machines(tmp_path, monkeypatch, v2, v1, meminfo, expected):
    _files(tmp_path, monkeypatch, v2, v1, meminfo)
    assert W.free_mb() == expected


# ---------- the work itself ----------

def test_the_warm_up_fills_the_entry_a_first_visit_reads(client, data):
    # the executive overview's own fixture: nothing cached, then the warm-up, then a visitor
    E._overview_cached.cache_clear()
    W.warm()
    info = E._overview_cached.cache_info()
    assert (info.currsize, info.misses) == (1, 1)
    # the page's first request, as the browser sends it (or with its defaults spelt out)
    for params in ({}, {"Plant": "ALL", "Month": AUG, "Basis": "value"}):
        assert client.get("/executive/overview", params=params).status_code == 200
    after = E._overview_cached.cache_info()
    assert (after.misses, after.hits - info.hits) == (1, 2)
