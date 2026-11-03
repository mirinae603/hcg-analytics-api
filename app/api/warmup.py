"""Cache warm-up: the overview a first visitor opens, computed in the background so they do not
wait for it.

The executive overview's first answer reads every section's tables and works out each one's
default view (sales, formulary, cash & TPA, scheme, out of formulary, prices): about 1 s of
work on the Oct-2026 tables, while every answer after it is about 1 ms from the result caches.
Those caches are empty after a start and after POST /admin/refresh-data, so both start a warm-up
(app/main.py's lifespan, dashboard_summary.refresh_data). Asking for the executive's default
scope fills the sections' own default views on the way, since it calls their handlers with the
same arguments the pages do.

It never blocks a start or a request, and never raises: it runs in a daemon thread, a failure is
logged and the next visitor fills the caches the usual way. It runs only when there is room for
it. The tables stay in memory once read (WARM_NEEDS_MB), and app/main.py's note on the 512 MB
free tier still holds: reading them at boot on a box that cannot hold them would crash it into a
restart loop, so with less memory free than that, nothing is warmed and the page loads on first
visit, as before.

It runs only in a serving process: the lifespan marks the start (serving()). The tests build
the app with TestClient outside `with`, which never runs the lifespan (tests/conftest.py), so
the refresh endpoint they call warms nothing and no thread reads the real tables under them.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

_log = logging.getLogger(__name__)
# The app configures no logging (as in app/ai/orchestrator.py): a plain stderr handler only when
# nobody has configured one, so the warm-up's one line per run is seen.
if not _log.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)

# Memory a warm-up needs free before it starts. On the Oct-2026 tables the executive's first
# answer took a 212 MB process to 685 MB at its peak and kept about 400 MB of it (the sales_items
# table, the spend tables and the purchase prices stay in their read caches). With room to spare.
WARM_NEEDS_MB = 600
# Where a container's memory limit and use are read: cgroup v2, then v1. With no limit set the
# machine's MemAvailable decides (MEMINFO).
CGROUP_FILES = (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
NO_LIMIT = 1 << 60            # cgroup v1 writes 'no limit' as a number near 2**63
MEMINFO = "/proc/meminfo"

_lock = threading.Lock()
_serving = False              # set by the lifespan: this process serves requests
_running = False              # a warm-up thread is at work
_again = False                # asked again while it ran: run once more for the newest data


def free_mb() -> Optional[float]:
    """Memory this process can still take, in MB: the container's limit less what it uses, else
    the machine's MemAvailable; None when neither can be read (then the warm-up runs)."""
    for limit_path, usage_path in CGROUP_FILES:
        try:
            limit = Path(limit_path).read_text().strip()
            usage = int(Path(usage_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if limit != "max" and limit.isdigit() and int(limit) < NO_LIMIT:
            return (int(limit) - usage) / 2 ** 20
        break                                   # a cgroup with no limit: the machine decides
    try:
        for line in Path(MEMINFO).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def warm() -> None:
    """The work itself: the executive overview's default scope (and so each section's default
    view it calls). Its answer goes to the result caches; nothing is returned."""
    from app.api import executive             # noqa: PLC0415 (loaded when first warmed, not at import)
    executive.executive_overview()


def serving() -> None:
    """Mark this process as a server (the lifespan's start): start() warms from now on."""
    global _serving
    with _lock:
        _serving = True


def start(reason: str) -> bool:
    """Warm the caches in a background thread. False when nothing starts: not a serving process,
    or a warm-up already at work (it then runs once more when done, so data rebuilt under it is
    warmed too). Never raises, never waits."""
    global _running, _again
    with _lock:
        if not _serving:
            return False
        if _running:
            _again = True
            return False
        _running = True
    try:
        threading.Thread(target=_run, args=(reason,), name="cache-warmup", daemon=True).start()
    except Exception:                           # no thread to be had: the first visitor waits instead
        with _lock:
            _running = False
        _log.warning("cache warm-up (%s) not started", reason, exc_info=True)
        return False
    return True


def _run(reason: str) -> None:
    global _running, _again
    while True:
        _warm_once(reason)
        with _lock:
            if not _again:
                _running = False
                return
            _again = False
        reason = "asked again"


def _warm_once(reason: str) -> None:
    try:
        free = free_mb()
        if free is not None and free < WARM_NEEDS_MB:
            _log.info("cache warm-up (%s) skipped: %.0f MB free, %d MB needed; the first visit fills the caches",
                      reason, free, WARM_NEEDS_MB)
            return
        t0 = time.perf_counter()
        warm()
        _log.info("cache warm-up (%s) done in %.1f s", reason, time.perf_counter() - t0)
    except HTTPException as e:                  # a table not built (503) is not a failure
        _log.info("cache warm-up (%s) skipped: %s", reason, e.detail)
    except Exception:                           # the next visitor fills the caches the usual way
        _log.warning("cache warm-up (%s) failed", reason, exc_info=True)
