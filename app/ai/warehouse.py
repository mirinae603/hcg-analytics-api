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


def _clean_group(g) -> str:
    g = str(g).strip()
    if not g or g.lower() in ("nan", "none"):
        return "Uncategorised"
    g = g.split("-", 1)[-1] if "-" in g else g
    return g.strip().title() or "Uncategorised"


def mentions(q: str, limit: int = 12) -> list[dict]:
    """Live entity search for the @-picker — items, vendors, manufacturers,
    categories, hospitals. Injection-safe (parameterised), read-only."""
    q = (q or "").strip()
    if not q:
        return []
    c = con()
    specs = [
        ("item", "SELECT DISTINCT material_desc FROM dim_material WHERE material_desc IS NOT NULL AND material_desc ILIKE '%' || ? || '%' ORDER BY length(material_desc), material_desc LIMIT 7", False),
        ("vendor", "SELECT DISTINCT vendor_name FROM dim_vendor WHERE vendor_name IS NOT NULL AND vendor_name ILIKE '%' || ? || '%' ORDER BY length(vendor_name), vendor_name LIMIT 5", False),
        ("manufacturer", "SELECT DISTINCT manufacturer_desc FROM dim_material WHERE manufacturer_desc IS NOT NULL AND manufacturer_desc != '' AND manufacturer_desc ILIKE '%' || ? || '%' ORDER BY length(manufacturer_desc) LIMIT 5", False),
        ("category", "SELECT DISTINCT material_group FROM dim_material WHERE material_group IS NOT NULL AND material_group ILIKE '%' || ? || '%' ORDER BY length(material_group) LIMIT 5", True),
        ("hospital", "SELECT DISTINCT hospital FROM sales_by_hospital WHERE hospital IS NOT NULL AND hospital ILIKE '%' || ? || '%' LIMIT 3", False),
    ]
    per_type: dict[str, list] = {}
    seen = set()
    with _lock:
        for typ, sql, is_cat in specs:
            try:
                rows = c.execute(sql, [q]).fetchall()
            except Exception:
                rows = []
            bucket = []
            for (label,) in rows:
                if label is None:
                    continue
                disp = _clean_group(label) if is_cat else str(label)
                key = (typ, disp.lower())
                if disp and key not in seen:
                    seen.add(key)
                    bucket.append({"type": typ, "label": disp})
            per_type[typ] = bucket
    # interleave (round-robin) so every category is represented, items first
    order = ["item", "category", "manufacturer", "vendor", "hospital"]
    out: list[dict] = []
    i = 0
    while len(out) < limit and any(per_type.get(t) and i < len(per_type[t]) for t in order):
        for t in order:
            b = per_type.get(t, [])
            if i < len(b) and len(out) < limit:
                out.append(b[i])
        i += 1
    return out


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
