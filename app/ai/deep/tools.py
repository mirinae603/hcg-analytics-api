"""The toolbox an analyst actually uses — as callable tools, not a static schema dump.

WHY
---
Asked for a sales trend, the model wrote `GROUP BY month` against `sales_by_material`
three times, on three tables whose column lists were in its own prompt, none of which has
a month column. It was not being stupid. It was GUESSING, because it had been handed a
description of the warehouse and then asked to produce a final query in one shot, blind.

A human does not do that. They open the table, look at five rows, notice there is no month,
go looking for one, find `mart_procurement` and `kpi_monthly_purchase_value`, realise no
monthly SALES grain exists, and answer with monthly PURCHASING while saying so. That is
eight actions with LOOKING between each one, and the intelligence is in the looking.

So: give it eyes. Every one of these mirrors something an analyst does before committing to
a query — list, describe, peek, find the grain, find where a value lives, profile a column,
try it, check it. The loop that drives them lives in engine.py.

Read-only throughout: every query goes through warehouse.validate() and the scope guards.
"""
from __future__ import annotations

import json
import re

from app.ai import warehouse
from app.ai.deep import capability

MAX_SAMPLE = 8
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe(name: str) -> bool:
    return bool(_IDENT.fullmatch(name or ""))


# ── the tools ────────────────────────────────────────────────────────────────
def list_tables(contains: str = "") -> dict:
    """What is in this warehouse (optionally filtered), with row counts."""
    sch = capability.schema()
    names = [t for t in sorted(sch) if not contains or contains.lower() in t.lower()]
    out = []
    for t in names[:60]:
        try:
            n = warehouse.run_sql(f"SELECT COUNT(*) AS n FROM {t}", row_cap=1)["rows"][0]["n"]
        except Exception:
            n = None
        out.append({"table": t, "rows": n, "columns": len(sch.get(t, []))})
    return {"tables": out, "total": len(names)}


def describe_table(table: str) -> dict:
    """Columns and types. The thing that should have been checked before GROUP BY month."""
    if not _safe(table):
        return {"error": "bad table name"}
    try:
        cols = warehouse.run_sql(
            "SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{table}' ORDER BY ordinal_position", row_cap=200)["rows"]
        n = warehouse.run_sql(f"SELECT COUNT(*) AS n FROM {table}", row_cap=1)["rows"][0]["n"]
    except Exception as e:
        return {"error": str(e)[:200]}
    if not cols:
        return {"error": f"no table named '{table}'"}
    return {"table": table, "rows": n,
            "columns": [{"name": c["column_name"], "type": c["data_type"]} for c in cols]}


def sample_rows(table: str, n: int = 5) -> dict:
    """Actual rows. Column names tell you a column exists; rows tell you what is IN it."""
    if not _safe(table):
        return {"error": "bad table name"}
    try:
        r = warehouse.run_sql(f"SELECT * FROM {table} LIMIT {min(int(n), MAX_SAMPLE)}", row_cap=MAX_SAMPLE)
        return {"table": table, "columns": r["columns"], "rows": r["rows"]}
    except Exception as e:
        return {"error": str(e)[:200]}


def find_columns(name_like: str) -> dict:
    """Which tables have a column like this — i.e. WHICH TABLES CARRY THIS GRAIN.

    "Who has a month?" is the question that would have prevented the whole KEYTRUDA
    failure, and it is one call.
    """
    q = (name_like or "").lower().strip()
    if not q:
        return {"error": "name_like is required"}
    hits = [{"table": t, "column": c}
            for t, cols in capability.schema().items() for c in cols if q in c.lower()]
    return {"looking_for": name_like, "found": hits[:60], "count": len(hits)}


def find_value(value: str) -> dict:
    """Which column actually CONTAINS this value.

    'Bangalore' is not a column, it is a value inside dim_plant.plant_name. Without this,
    the model concluded the warehouse had no location data at all.
    """
    v = (value or "").strip()
    if len(v) < 2:
        return {"error": "value too short"}
    lit = v.replace("'", "''")
    hits = []
    for table, col in capability.entity_columns_for_search():
        try:
            n = warehouse.run_sql(
                f"SELECT COUNT(*) AS n FROM {table} WHERE upper(CAST({col} AS VARCHAR)) "
                f"LIKE '%' || upper('{lit}') || '%'", row_cap=1, timeout_s=3.0)["rows"][0]["n"]
        except Exception:
            continue
        if n:
            hits.append({"table": table, "column": col, "rows": int(n)})
        if len(hits) >= 8:
            break
    hits.sort(key=lambda h: -h["rows"])
    return {"value": v, "found_in": hits, "note": "" if hits else "not found in any entity column"}


def profile_column(table: str, column: str) -> dict:
    """Distinct count, range, nulls, and the commonest values — is this usable as a filter?"""
    if not (_safe(table) and _safe(column)):
        return {"error": "bad identifier"}
    try:
        base = warehouse.run_sql(
            f"SELECT COUNT(*) AS n, COUNT({column}) AS non_null, "
            f"COUNT(DISTINCT {column}) AS distinct_n FROM {table}", row_cap=1)["rows"][0]
        top = warehouse.run_sql(
            f"SELECT CAST({column} AS VARCHAR) AS value, COUNT(*) AS n FROM {table} "
            f"WHERE {column} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 8", row_cap=8)["rows"]
    except Exception as e:
        return {"error": str(e)[:200]}
    return {"table": table, "column": column, **base, "most_common": top}


_COUNTING_WORDS = re.compile(r"\bhow many\b|\bnumber of\b|\bcount of\b", re.I)
_ITEM_WORDS = re.compile(r"\bskus?\b|\bitems?\b|\bproducts?\b|\bdrugs?\b|\bmaterials?\b|\bmedicines?\b", re.I)
_COUNT_STAR = re.compile(r"count\s*\(\s*\*\s*\)", re.I)


def _miscounts_items(sql: str, question: str) -> str | None:
    """COUNT(*) is not an item count on a table keyed by item PER SITE.

    kpi_non_moving holds 16,872 rows and 10,501 distinct materials — one row per material
    per hospital. Asked how many SKUs are non-moving, the answer was 16,872. The grain is
    stated in the lesson board and was read past, which is what a stated preference is
    worth; this makes it a property of the query instead.
    """
    q = question or ""
    if not (_COUNTING_WORDS.search(q) and _ITEM_WORDS.search(q)):
        return None
    if not _COUNT_STAR.search(sql or ""):
        return None
    from app.ai.deep import capability
    low = (sql or "").lower()
    for note in capability.grain_notes():
        table = note.split(":", 1)[0]
        if re.search(rf"\b{re.escape(table)}\b", low):
            return (f"COUNT(*) on {table} counts rows, and {note.split('—', 1)[-1].strip()} "
                    f"The question asks how many ITEMS, so use COUNT(DISTINCT material).")
    return None


def run_query(sql: str, entity_tokens: list[str] | None = None, question: str = "") -> dict:
    """Run it and SEE the result — including the error, which is information, not a dead end."""
    from app.ai import scope
    miscount = _miscounts_items(sql, question)
    if miscount:
        return {"error": miscount}
    off = scope.missing_entity_scope(sql, entity_tokens or [])
    if off:
        return {"error": off}
    # A city filter on a sales table cannot be satisfied — the code systems are disjoint.
    # Saying so in the prompt did not stop it; failing the query does.
    from app.ai.deep import sanity as _sanity
    unreachable = _sanity.city_on_unreachable_table(sql)
    if unreachable:
        return {"error": unreachable}
    try:
        r = warehouse.run_sql(sql, row_cap=200)
    except Exception as e:
        return {"error": str(e)[:300]}
    # `ORDER BY spend DESC LIMIT 1` returning "Uncategorized" cannot be repaired by
    # reordering — there is nothing left to promote. Send it back to be re-queried.
    unnamed = _sanity.placeholder_won_a_ranking(sql, r, question)
    if unnamed:
        return {"error": unnamed}
    # SUM() over zero matching rows returns ONE row containing NULL, which looks like a
    # result and is not: "SELECT SUM(revenue) FROM sales_by_manufacturer WHERE manufacturer
    # = 'MSD'" matched nothing (the value is stored 'Msd') and the answer became "there is
    # no sales figure for MSD" for a manufacturer with ₹47.57 Cr.
    rows = r.get("rows") or []
    if r.get("row_count") == 1 and rows and all(v is None for v in rows[0].values()):
        hint = scope.explain_zero_rows(sql)
        return {"row_count": 0, "columns": r["columns"],
                "note": (hint or "") + " Every value came back NULL, which means the filter "
                        "matched no rows — check the spelling and CASE of the value you "
                        "filtered on (use upper()/ILIKE), not the aggregate."}
    if not r.get("row_count"):
        hint = scope.explain_zero_rows(sql)
        return {"row_count": 0, "columns": r["columns"],
                "note": hint or "0 rows — check the filter values with find_value or profile_column"}
    return {"row_count": r["row_count"], "columns": r["columns"], "rows": r["rows"][:12],
            "truncated": r["row_count"] > 12, "_full": r}


def get_kpi(key: str, plant: str = "", category: str = "") -> dict:
    """The dashboard's OWN calculation for a named metric — correct by construction.

    Deep mode was re-deriving everything with hand-written SQL, including metrics the
    application already computes exactly. Asked how much stock expires in 90 days it
    returned "45,223 units" while the canonical KPI — the number on the dashboard card —
    is ₹39.97 L across 45,223 units and 869 items. Not wrong, but re-derived when a
    verified answer was one call away. Every metric taken from here is one fewer LLM
    decision in the chain, which is the only thing that actually compounds reliability.
    """
    from app.ai import kpi_registry
    if key not in kpi_registry.KPI_REGISTRY:
        return {"error": f"unknown kpi '{key}'",
                "available": sorted(kpi_registry.KPI_REGISTRY)[:40]}
    try:
        payload = kpi_registry.call_kpi(key, plant or None, category or None)
    except Exception as e:
        return {"error": str(e)[:220]}
    return {"kpi": key, "canonical": True, "payload": payload}


def kpi_keys() -> list[str]:
    from app.ai import kpi_registry
    return sorted(kpi_registry.KPI_REGISTRY)


# ── OpenAI tool schemas ──────────────────────────────────────────────────────
def _t(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


SPECS = [
    _t("list_tables", "List warehouse tables with row counts. Start here if unsure what exists.",
       {"contains": {"type": "string", "description": "optional substring filter"}}, []),
    _t("describe_table", "Columns and types of one table. ALWAYS call this before writing a "
       "GROUP BY or WHERE on a column you have not already seen in this conversation.",
       {"table": {"type": "string"}}, ["table"]),
    _t("sample_rows", "A few real rows, so you can see what the values look like.",
       {"table": {"type": "string"}, "n": {"type": "integer"}}, ["table"]),
    _t("find_columns", "Which tables have a column whose name contains this. Use it to find "
       "which tables carry a grain — e.g. find_columns('month') before assuming one has months.",
       {"name_like": {"type": "string"}}, ["name_like"]),
    _t("find_value", "Which table.column actually CONTAINS this value (e.g. 'Bangalore', 'MSD'). "
       "Use it when a filter returns nothing, or to locate a name you were given.",
       {"value": {"type": "string"}}, ["value"]),
    _t("profile_column", "Distinct count, nulls and commonest values of a column.",
       {"table": {"type": "string"}, "column": {"type": "string"}}, ["table", "column"]),
    _t("get_kpi", "The dashboard's OWN calculation for a named metric — correct by "
       "construction and already verified. PREFER THIS over writing SQL whenever the "
       "question matches one of these metrics; only fall back to run_query for things no "
       "KPI covers.",
       {"key": {"type": "string", "enum": kpi_keys()},
        "plant": {"type": "string", "description": "hospital code, or empty for all"},
        "category": {"type": "string"}}, ["key"]),
    _t("run_query", "Run a read-only SELECT and see the rows.",
       {"sql": {"type": "string"}, "purpose": {"type": "string"}}, ["sql", "purpose"]),
    _t("finish", "Call this once you have the rows that answer the sub-question. Pass the SQL "
       "that produced them.",
       {"sql": {"type": "string"}, "what_it_shows": {"type": "string"}}, ["sql", "what_it_shows"]),
    _t("give_up", "Call this only after LOOKING — you have checked which tables carry the grain "
       "and the entity and none does. Say precisely what is missing.",
       {"reason": {"type": "string"}, "closest_available": {"type": "string"}}, ["reason"]),
]

DISPATCH = {
    "list_tables": lambda a, ctx: list_tables(a.get("contains", "")),
    "describe_table": lambda a, ctx: describe_table(a.get("table", "")),
    "sample_rows": lambda a, ctx: sample_rows(a.get("table", ""), a.get("n", 5)),
    "find_columns": lambda a, ctx: find_columns(a.get("name_like", "")),
    "find_value": lambda a, ctx: find_value(a.get("value", "")),
    "profile_column": lambda a, ctx: profile_column(a.get("table", ""), a.get("column", "")),
    "get_kpi": lambda a, ctx: get_kpi(a.get("key", ""), a.get("plant", ""), a.get("category", "")),
    "run_query": lambda a, ctx: run_query(a.get("sql", ""), ctx.get("entity_tokens"), ctx.get("question", "")),
}


def call(name: str, args: dict, ctx: dict) -> dict:
    fn = DISPATCH.get(name)
    if not fn:
        return {"error": f"no tool named {name}"}
    try:
        return fn(args, ctx)
    except Exception as e:  # a tool must never kill the loop
        return {"error": str(e)[:200]}


def compact(obj: dict, limit: int = 2600) -> str:
    o = {k: v for k, v in obj.items() if not k.startswith("_")}
    s = json.dumps(o, default=str)
    return s if len(s) <= limit else s[:limit] + " …(truncated)"
