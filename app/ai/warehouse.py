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


# Reserved-word column names → safe, schema-consistent aliases. A column literally
# named `desc` or `group` (both SQL keywords) forces the LLM to quote it, which it
# reliably forgets — producing a ParserException the user sees as a "technical issue".
# We alias them away at load time so plain `SELECT material_desc FROM …` just works.
_RENAME = {"desc": "material_desc", "group": "material_group"}


def _needs_alias(con: duckdb.DuckDBPyConnection, view: str, col: str) -> bool:
    try:
        con.execute(f"SELECT {col} FROM {view} LIMIT 0")
        return False
    except Exception:
        return True   # reserved keyword / otherwise unquotable as a bare identifier


def _harden_view(con: duckdb.DuckDBPyConnection, name: str, path: str) -> None:
    """Recreate the view with any reserved-word column aliased to a safe name, so the
    agent never has to quote identifiers (quoting is what it forgets)."""
    cols = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
    lower = {c.lower() for c in cols}
    renames: dict[str, str] = {}
    for c in cols:
        if not _needs_alias(con, name, c):
            continue
        tgt = _RENAME.get(c.lower(), f"{c}_col")
        if tgt.lower() in lower and tgt.lower() != c.lower():
            tgt = f"{c}_col"
        if tgt[:1].isdigit():
            tgt = "c_" + tgt
        renames[c] = tgt
    if not renames:
        return
    proj = ", ".join(f'"{c}" AS {renames[c]}' if c in renames else f'"{c}"' for c in cols)
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT {proj} FROM read_parquet('{path}')")


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
                _harden_view(con, name, f)
                _tables[name] = f
            except Exception:
                pass
    return con


# Confirmed source-data defect: these 3 sales_by_material* views store the SAME
# product under TWO material-key formats ('218766' and '218766.0') as fully
# separate rows, splitting its revenue/cost/qty across them — 33-45% of distinct
# materials, up to 85% of total revenue by value in sales_by_material. Verified
# by direct inspection: whenever a duplicate pair exists, the '.0'-suffixed row
# is always the one missing material_group (blank) — never a genuine second SKU
# or a real category conflict (0 cases of two different non-blank groups/descs
# for the same normalized material across all three views). Fixed once here, at
# the view layer, so every consumer (agent SQL, entity picker, deterministic KPI
# code) sees one correct row per product no matter how the question is phrased,
# instead of depending on every query remembering to re-aggregate by material.
_MATERIAL_DUP_VIEWS = {
    "sales_by_material": [],
    "sales_by_material_hospital": ["hospital"],
    "sales_by_material_mfr": ["manufacturer"],
}


def _fix_material_duplicate_keys(con: duckdb.DuckDBPyConnection) -> None:
    for name, extra_dims in _MATERIAL_DUP_VIEWS.items():
        path = _tables.get(name)
        if not path:
            continue
        cols = "".join(f'"{d}", ' for d in extra_dims)
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT {cols}
                   mat_norm AS material,
                   arg_max("desc", length("desc")) AS material_desc,
                   COALESCE(max(NULLIF("group", '')), '') AS material_group,
                   sum(revenue) AS revenue, sum(cost) AS cost, sum(qty) AS qty, sum(lines) AS lines
            FROM (
                SELECT *, CASE WHEN material LIKE '%.0' THEN LEFT(material, LENGTH(material) - 2)
                               ELSE material END AS mat_norm
                FROM read_parquet('{path}')
            )
            GROUP BY {cols}mat_norm
        """)


# ── Dedicated CHATBOT MART (procurement / pricing / vendor) ──────────────────
# Many confirmed AI-analyst bugs in this domain were NOT reasoning failures — they
# were the model having to re-derive the same cleaning/joins from scratch every turn
# and doing it inconsistently: skipping the outlier-clean two-pass on a price ranking,
# comparing raw SUM(spend) across plants that hold different quantities, mis-joining
# item→vendor lead time (kpi_vendor_lead_time has NO material column), or reading the
# raw messy category. This mart bakes ALL of that in as DATA so the model just reads
# one already-correct column:
#   • outlier-cleaned median price per material (net_price>=10, drop rows outside an
#     8x band of the material's own median — gotcha #15's rule, computed ONCE) + the
#     sample size backing it, so "is this price stable enough to rank?" is a column.
#   • canonical category (dim_material.material_group), never the raw messy field.
#   • is_capex_or_service flag (material NULL) so a construction/service line can't be
#     surfaced as "the top item purchased".
#   • unit_price per line so cross-plant/vendor price comparison is per-UNIT by default.
#   • vendor lead time pre-joined onto every line (removes the 2-hop join the model
#     kept getting wrong).
# The two small stat tables are MATERIALIZED (the median computation runs once, a few
# MB total); the big line-level view stays a zero-memory VIEW that joins to them — so
# nothing here meaningfully grows RSS against the 512MB tier.
_MART_NAMES = ("mart_material_price_stats", "mart_material_vendor_price_stats", "mart_procurement")


def _build_procurement_mart(con: duckdb.DuckDBPyConnection) -> None:
    # 1. material-grain clean price stats (materialized — the outlier-clean two-pass, once)
    con.execute("""
        CREATE OR REPLACE TABLE mart_material_price_stats AS
        WITH raw AS (
            SELECT material, net_price FROM fact_grn
            WHERE material IS NOT NULL AND net_price >= 10 AND gr_qty > 0
        ),
        p1 AS (SELECT material, median(net_price) AS m1 FROM raw GROUP BY 1),
        clean AS (
            SELECT r.material, r.net_price FROM raw r JOIN p1 p USING(material)
            WHERE r.net_price BETWEEN p.m1 / 8.0 AND p.m1 * 8.0
        ),
        rawstat AS (SELECT material, count(*) AS raw_n, min(net_price) AS raw_min_price,
                           max(net_price) AS raw_max_price FROM raw GROUP BY 1),
        cleanstat AS (SELECT material, median(net_price) AS clean_median_price,
                             count(*) AS clean_n, min(net_price) AS clean_min_price,
                             max(net_price) AS clean_max_price FROM clean GROUP BY 1)
        SELECT a.material, dm.material_desc, dm.material_group AS category,
               cs.clean_median_price, COALESCE(cs.clean_n, 0) AS clean_n,
               cs.clean_min_price, cs.clean_max_price,
               -- genuine price dispersion within the clean band (real over/under-pay signal,
               -- NOT the raw transposition noise). Use THIS for "are we overpaying", not the
               -- 20x has_price_outlier flag (that catches only gross data-entry errors).
               (cs.clean_max_price - cs.clean_min_price) AS clean_price_spread,
               a.raw_n, a.raw_min_price, a.raw_max_price,
               (a.raw_max_price > a.raw_min_price * 20) AS has_price_outlier,
               (COALESCE(cs.clean_n, 0) >= 5) AS price_is_stable
        FROM rawstat a
        LEFT JOIN cleanstat cs USING(material)
        LEFT JOIN dim_material dm ON a.material = dm.material
    """)
    # 2. material x vendor clean price stats (materialized) — "who is cheapest for item X"
    con.execute("""
        CREATE OR REPLACE TABLE mart_material_vendor_price_stats AS
        WITH raw AS (
            SELECT material, vendor_name, net_price, gr_qty, total_amount_wo_tax FROM fact_grn
            WHERE material IS NOT NULL AND net_price >= 10 AND gr_qty > 0
        ),
        p1 AS (SELECT material, vendor_name, median(net_price) AS m1 FROM raw GROUP BY 1, 2),
        clean AS (
            SELECT r.* FROM raw r JOIN p1 p ON r.material = p.material AND r.vendor_name = p.vendor_name
            WHERE r.net_price BETWEEN p.m1 / 8.0 AND p.m1 * 8.0
        ),
        st AS (
            SELECT material, vendor_name, median(net_price) AS clean_median_price, count(*) AS clean_n,
                   sum(gr_qty) AS total_qty, sum(total_amount_wo_tax) AS total_value
            FROM clean GROUP BY 1, 2
        )
        SELECT st.material, dm.material_desc, dm.material_group AS category,
               st.vendor_name, st.clean_median_price, st.clean_n,
               st.total_qty, st.total_value, (st.clean_n >= 5) AS price_is_stable
        FROM st LEFT JOIN dim_material dm ON st.material = dm.material
    """)
    # 3. line-grain enriched procurement view (zero extra memory — joins the small tables above)
    con.execute("""
        CREATE OR REPLACE VIEW mart_procurement AS
        SELECT g.plant, g.material, g.material_desc,
               dm.material_group AS category, dm.material_type, dm.manufacturer_desc, dm.generic_name,
               g.vendor_code, g.vendor_name, g.po_no, g.gr_no,
               g.gr_qty AS qty,
               g.net_price AS unit_price,
               g.unit_mrp,
               g.total_amount_wo_tax AS line_value,
               (g.material IS NULL) AS is_capex_or_service,
               ps.clean_median_price AS material_median_price,
               ps.clean_n AS material_price_sample_n,
               (ps.clean_median_price IS NOT NULL
                 AND (g.net_price < ps.clean_median_price / 8.0 OR g.net_price > ps.clean_median_price * 8.0)
               ) AS is_price_outlier,
               lt.avg_lead_time_days AS vendor_avg_lead_time_days,
               lt.median_lead_time_days AS vendor_median_lead_time_days,
               g.gr_date, g.expiry_date, g.year, g.month_num, g.month
        FROM fact_grn g
        LEFT JOIN dim_material dm ON g.material = dm.material
        LEFT JOIN mart_material_price_stats ps ON g.material = ps.material
        LEFT JOIN kpi_vendor_lead_time lt ON g.vendor_name = lt.vendor_name
    """)
    for name in _MART_NAMES:
        _tables[name] = ""   # register so schema_text()/grain_text() expose them to the model


def con() -> duckdb.DuckDBPyConnection:
    global _con
    with _lock:
        if _con is None:
            _con = _connect()
            _fix_material_duplicate_keys(_con)
            _build_procurement_mart(_con)
    return _con


def tables() -> list[str]:
    con()
    return sorted(_tables.keys())


def _coerce(v):
    if v is None:
        return None
    if isinstance(v, Decimal):
        v = float(v)
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()[:10]
    if isinstance(v, float):
        # A 0/0 ratio (e.g. margin_pct for a manufacturer with zero revenue AND zero
        # cost) produces NaN/Infinity — a raw float artifact, not a real number. Never
        # let it reach the model or the answer as "NaN%"; treat it as "not computable".
        if v != v or v in (float("inf"), float("-inf")):
            return None
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


_MEN_TYPES = ["item", "category", "manufacturer", "vendor", "hospital"]


def _men_spec(t: str):
    # (table, column, browse_sql[l, ord], is_category)  — browse is ordered by 'ord' desc (importance)
    return {
        "item": ("sales_by_material", "material_desc", "SELECT material_desc AS l, revenue AS ord FROM sales_by_material", False),
        "vendor": ("dim_vendor", "vendor_name", "SELECT vendor_name AS l, sum(vendor_value) AS ord FROM kpi_vendor_volume GROUP BY 1", False),
        "manufacturer": ("dim_material", "manufacturer_desc", "SELECT manufacturer AS l, revenue AS ord FROM sales_by_manufacturer", False),
        "category": ("dim_material", "material_group", "SELECT material_group AS l, count(*) AS ord FROM dim_material GROUP BY 1", True),
        "hospital": ("sales_by_hospital", "hospital", "SELECT hospital AS l, revenue AS ord FROM sales_by_hospital", False),
    }[t]


def mentions(q: str, mtype: str | None = None, limit: int = 18) -> list[dict]:
    """Entity picker source. With a query → ILIKE search (prefix-ranked). Empty query
    → a CURATED browse list ordered by importance (top items by revenue, top vendors by
    spend, biggest categories, etc.). Injection-safe (parameterised), read-only."""
    q = (q or "").strip()
    c = con()
    types = [mtype] if mtype in _MEN_TYPES else _MEN_TYPES
    cap = limit if mtype in _MEN_TYPES else 6   # per-type cap for the cross-type mix
    per: dict[str, list] = {}
    with _lock:
        for t in types:
            tbl, col, browse, is_cat = _men_spec(t)
            try:
                if q:
                    sql = (f"SELECT DISTINCT {col} AS l FROM {tbl} "
                           f"WHERE {col} IS NOT NULL AND {col} != '' AND {col} ILIKE '%' || ? || '%' "
                           f"ORDER BY (CASE WHEN {col} ILIKE ? || '%' THEN 0 ELSE 1 END), length({col}), {col} LIMIT ?")
                    rows = c.execute(sql, [q, q, cap]).fetchall()
                else:
                    rows = c.execute(f"SELECT l FROM ({browse}) t WHERE l IS NOT NULL AND l != '' ORDER BY ord DESC NULLS LAST LIMIT ?", [cap]).fetchall()
            except Exception:
                rows = []
            seen = set(); bucket = []
            for (label,) in rows:
                if label is None:
                    continue
                disp = _clean_group(label) if is_cat else str(label)
                k = disp.lower()
                if disp and k not in seen:
                    seen.add(k)
                    bucket.append({"type": t, "label": disp})
            per[t] = bucket
    if mtype in _MEN_TYPES:
        return per.get(mtype, [])[:limit]
    # cross-type: interleave (round-robin) so each type shows up
    out: list[dict] = []; i = 0
    while len(out) < limit and any(i < len(per.get(t, [])) for t in types):
        for t in types:
            b = per.get(t, [])
            if i < len(b) and len(out) < limit:
                out.append(b[i])
        i += 1
    return out


_TYPE_SHORT = {
    "VARCHAR": "text", "TEXT": "text", "CHAR": "text", "STRING": "text",
    "DOUBLE": "num", "FLOAT": "num", "REAL": "num", "DECIMAL": "num",
    "BIGINT": "int", "INTEGER": "int", "HUGEINT": "int", "SMALLINT": "int", "TINYINT": "int", "UBIGINT": "int",
    "TIMESTAMP": "date", "DATE": "date", "TIME": "date", "TIMESTAMP_NS": "date",
    "BOOLEAN": "bool", "BOOL": "bool",
}


def _short_type(t: str) -> str:
    base = str(t).split("(")[0].strip().upper()
    return _TYPE_SHORT.get(base, base.lower())


def schema_text() -> str:
    """Compact TYPED schema listing for the agent prompt so it knows exactly what
    every table holds: `table [N rows]: col:type, col:type, …`."""
    c = con()
    lines = []
    for name in tables():
        try:
            info = c.execute(f"DESCRIBE {name}").fetchall()
            n = c.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            cols = ", ".join(f"{r[0]}:{_short_type(r[1])}" for r in info)
            lines.append(f"{name} [{n:,} rows]: {cols}")
        except Exception:
            pass
    return "\n".join(lines)


# ── Auto-derived dimensional model (grain) ───────────────────────────────────
# Every column is mechanically classified as a TIME axis, a DIMENSION (something you
# can group/filter by), or a MEASURE (a number you aggregate). The agent is then told
# exactly what each table can be sliced by — so it can't attribute a table's numbers to
# a dimension the table doesn't carry, or invent a time trend a table doesn't have.
_TIME_RE = re.compile(r"(^month$|^month_num$|^year$|^week$|^quarter$|_date$|^date$|snapshot|posting|^period$)", re.I)
_ID_RE = re.compile(r"(^material$|^material_id$|^plant$|^sloc$|^hsn$|^batch$|vendor_code|vendor_name|cost_ctr|^hospital$|^manufacturer$|manufacturer_desc|^patient$|po_no|gr_no|^generic_name$|_code$|_id$|_no$)", re.I)
_MEASURE_RE = re.compile(r"(revenue|cost|value|price|amount|margin|mrp|spend|overpay|opportunity|"
                         r"qty|quantity|units|unit_|count|lines|sku_count|"
                         r"pct|percent|share|rate|score|ratio|"
                         r"days|doh|aging|lead|tat|cover|coverage|"
                         r"stock|closing|demand|forecast|replenish|turnover|fulfil|fill|consumption|cashflow)", re.I)


def _classify_col(name: str, dtype: str) -> str:
    n = name.lower()
    short = _short_type(dtype)
    if _TIME_RE.search(n):
        return "time"
    if short in ("date",):
        return "time"
    if short == "text":
        return "dim"          # any label / code / name
    if _ID_RE.search(n):
        return "dim"          # numeric identifier
    if _MEASURE_RE.search(n):
        return "measure"
    return "measure"          # unknown numeric → treat as a measure


def grain_text() -> str:
    """One line per table describing its GRAIN: which dimensions it can be sliced by
    and whether it has a time axis. Derived purely from column names/types — general,
    no hand-coded per-table rules."""
    c = con()
    lines = []
    for name in tables():
        try:
            info = c.execute(f"DESCRIBE {name}").fetchall()
        except Exception:
            continue
        dims, times = [], []
        for r in info:
            col, dtype = r[0], r[1]
            k = _classify_col(col, dtype)
            if k == "time":
                times.append(col)
            elif k == "dim":
                dims.append(col)
        slice_by = ", ".join(dims) if dims else "— (single total row / no dimensions)"
        tpart = ", ".join(times) if times else "NONE — this is a fixed total, it has NO time axis"
        lines.append(f"{name}: slice by [{slice_by}] · time axis [{tpart}]")
    return "\n".join(lines)


def _material_col(cols: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in ("material", "material_id"):
        if cand in low:
            return low[cand]
    return None


def _code_variants(codes: list[str]) -> list[str]:
    """Some fact tables store the material key as a float string ('218766.0') while
    others use the int form ('218766'). Match both so a footprint never undercounts."""
    out: set[str] = set()
    for c in codes:
        s = str(c)
        out.add(s)
        if s.endswith(".0"):
            out.add(s[:-2])
        elif s.replace(".", "", 1).isdigit() and "." not in s:
            out.add(s + ".0")
    return list(out)


def item_footprint(name: str, limit: int = 8) -> dict:
    """Resolve a specific product (by material_desc name OR material code) and report
    its identity + a COMPLETE footprint: how many rows it has in EVERY table that keys
    on material. This is what stops the agent concluding "no data" after checking only
    sales/purchase — an item with 0 sales but rows in fact_inventory is dead stock, and
    this surfaces that in one deterministic call. Read-only, parameterised."""
    q = (name or "").strip()
    c = con()
    cols_m = ["material", "material_desc", "material_group", "generic_name",
              "manufacturer_desc", "formulary", "material_type"]
    with _lock:
        try:
            matches = c.execute(
                f"SELECT {', '.join(cols_m)} FROM dim_material "
                "WHERE material = ? OR material_desc ILIKE '%' || ? || '%' "
                "ORDER BY (CASE WHEN material = ? THEN 0 WHEN material_desc ILIKE ? || '%' THEN 1 ELSE 2 END), length(material_desc) "
                "LIMIT ?", [q, q, q, q, limit]).fetchall()
        except Exception:
            matches = []
        resolved = [{k: _coerce(v) for k, v in zip(cols_m, r)} for r in matches]
        codes = _code_variants([r["material"] for r in resolved if r.get("material")])
        footprint: dict[str, int] = {}
        if codes:
            ph = ",".join(["?"] * len(codes))
            for name_ in sorted(_tables.keys()):
                if name_.startswith("mart_"):
                    continue   # marts are derived from base facts — don't double-count the footprint
                try:
                    info = c.execute(f"DESCRIBE {name_}").fetchall()
                    mcol = _material_col([r[0] for r in info])
                    if not mcol:
                        continue
                    n = c.execute(f"SELECT count(*) FROM {name_} WHERE CAST({mcol} AS VARCHAR) IN ({ph})", codes).fetchone()[0]
                    footprint[name_] = int(n)
                except Exception:
                    pass
    return {
        "query": q,
        "match_count": len(resolved),
        "matches": resolved,
        "footprint": footprint,
        "tables_with_data": sorted([t for t, n in footprint.items() if n > 0]),
        "note": ("No catalog match — try a looser name or ask about the brand family."
                 if not resolved else
                 "footprint = row count per table for the matched material code(s); 0 means absent from that table."),
    }
