"""
In-process DuckDB warehouse over the parquet lake. Every KPI aggregate, fact and
dimension table is exposed as a read-only VIEW, so the AI analyst can write full
analytical SQL (joins, CTEs, window functions) across ALL of them.

Governance (why this is safe, unlike exec'ing LLM Python):
  • single-statement SELECT/WITH only — INSERT/UPDATE/DROP/ATTACH/COPY/PRAGMA/… rejected
  • hard row cap wrapped around every query
  • wall-clock timeout via con.interrupt() watchdog
  • the connection never writes to disk (:memory:, views over read_parquet)
"""
from __future__ import annotations
import datetime as _dt
import glob
import os
import re
import threading
from decimal import Decimal

import duckdb

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRS = [os.path.join(_HERE, "data", "kpi"), os.path.join(_HERE, "data", "curated")]

_con: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()
_tables: dict[str, str] = {}   # view name → parquet path

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|"
    r"load|export|import|call|set|reset|vacuum|checkpoint|grant|revoke|"
    r"read_csv|read_json|glob)\b", re.I)
_ALLOWED_START = re.compile(r"^\s*(with|select)\b", re.I)


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET threads TO 4")
    except Exception:
        pass
    for d in _DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            name = os.path.basename(f)[:-len(".parquet")]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                continue
            try:
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{f}')")
                _tables[name] = f
            except Exception:
                pass
    return con


def con() -> duckdb.DuckDBPyConnection:
    global _con
    with _lock:
        if _con is None:
            _con = _connect()
    return _con


def tables() -> list[str]:
    con()
    return sorted(_tables.keys())


def _coerce(v):
    if v is None:
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()[:10]
    if isinstance(v, float):
        return round(v, 4)
    return v


class SqlError(Exception):
    pass


def validate(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise SqlError("Empty query.")
    if ";" in s:
        raise SqlError("Only a single statement is allowed (no ';').")
    if not _ALLOWED_START.match(s):
        raise SqlError("Only read-only SELECT / WITH queries are allowed.")
    bad = _FORBIDDEN.search(s)
    if bad:
        raise SqlError(f"Disallowed keyword '{bad.group(0)}' — this analyst is read-only.")
    return s


def run_sql(sql: str, row_cap: int = 500, timeout_s: float = 20.0) -> dict:
    """Execute a governed read-only query. Returns {columns, rows, row_count, truncated, sql}."""
    s = validate(sql)
    wrapped = f"SELECT * FROM (\n{s}\n) AS _q LIMIT {int(row_cap) + 1}"
    c = con()
    with _lock:
        timer = threading.Timer(timeout_s, c.interrupt)
        timer.start()
        try:
            cur = c.execute(wrapped)
            colnames = [d[0] for d in cur.description]
            raw = cur.fetchall()
        except duckdb.InterruptException:
            raise SqlError(f"Query exceeded {timeout_s:.0f}s and was cancelled — narrow it down.")
        except Exception as e:
            raise SqlError(str(e).split("\n")[0][:300])
        finally:
            timer.cancel()
    truncated = len(raw) > row_cap
    raw = raw[:row_cap]
    rows = [{colnames[i]: _coerce(v) for i, v in enumerate(r)} for r in raw]
    return {"columns": colnames, "rows": rows, "row_count": len(rows), "truncated": truncated, "sql": s}


def schema_text() -> str:
    """Compact schema listing for the agent prompt: table[rows]: col:type, …"""
    c = con()
    lines = []
    for name in tables():
        try:
            info = c.execute(f"DESCRIBE {name}").fetchall()
            n = c.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            cols = ", ".join(f"{r[0]}" for r in info)
            lines.append(f"{name} [{n:,} rows]: {cols}")
        except Exception:
            pass
    return "\n".join(lines)
