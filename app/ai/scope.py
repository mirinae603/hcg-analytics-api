"""Scope enforcement for agent-written SQL.

WHY THIS EXISTS
---------------
Three answers, all wrong, all from one afternoon, all the same defect:

  "sales trend of KEYTRUDA"      -> ₹89.52 Cr for Dec-25. That is `sales_monthly`'s
                                    Dec-25 total for EVERY material. KEYTRUDA's real
                                    revenue across all six months is ₹47.48 Cr. The
                                    table has no `material` column, so the filter was
                                    silently dropped and company-wide numbers were
                                    presented as one product's trend.
  "procurement details for MSD"  -> "no recorded procurement details". There are 614
                                    lines worth ₹42.34 Cr. It filtered `vendor_name`;
                                    MSD lives in `manufacturer_desc`.
  "lead times for MSD supplies"  -> "no recorded lead times". Lead times are keyed by
                                    VENDOR and MSD is a MANUFACTURER; its suppliers
                                    (Vardhman, Arnav, Neha) all have lead times.

The common cause is not three bad queries. It is that nothing in the pipeline ever
checks that a query is actually SCOPED TO THE THING THAT WAS ASKED ABOUT:

  * `validate()` checks syntax and safety, not meaning.
  * `_unsupported_numbers` checks the prose against the query's own output — and the
    output of a wrongly-scoped query is internally consistent, so it passes.
  * The LLM auditor is documented elsewhere in this codebase as miscalibrated: it
    fires on correct answers and stays silent on real errors.

So the model picks a table by name similarity and, when the filter cannot be expressed
there, it drops it, fabricates the dimension, or filters the wrong column and reports
the empty result as fact. Every one of those is silent.

WHAT THIS MODULE ADDS
---------------------
Two deterministic checks, both general rather than per-question:

  1. `missing_entity_scope()` — a query run after an entity was resolved must mention
     that entity. Kills "asked about one product, answered about all of them" for any
     entity type, present or future.

  2. `explain_zero_rows()` — when a filtered query returns nothing, look for the
     filtered value elsewhere in the warehouse. If it lives in a different column, the
     answer is "wrong dimension", not "no data". Kills the MSD class of false negative.

Both are cheap, both fail closed into an explanatory message the model can act on, and
neither hard-codes a table, a column or a question.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Columns that identify a business entity rather than measure one. Anything matching
# these is a candidate scope key AND a candidate home for a mis-filtered literal.
_ENTITY_COL_RE = re.compile(
    r"(?:^|_)(material|material_desc|generic_name|manufacturer|manufacturer_desc|vendor|"
    r"vendor_name|vendor_code|plant|hospital|patient|category|material_group|"
    r"major_group_desc|minor_group_desc|material_type|department|formulary)(?:$|_)",
    re.I,
)

# Literals that carry no scoping meaning — filtering on these tells us nothing about
# which entity a query is about, so they must never satisfy a scope requirement or
# trigger a "wrong dimension" hunt.
_NOISE = {
    "y", "n", "yes", "no", "true", "false", "null", "none", "all", "total", "nan",
    "both", "internal", "billed", "other", "others", "unknown", "na",
}

_STR_LIT_RE = re.compile(r"'((?:[^']|'')*)'")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


@lru_cache(maxsize=1)
def entity_columns() -> tuple[tuple[str, str], ...]:
    """Every (table, column) in the warehouse that names an entity.

    Discovered from information_schema rather than listed by hand, so a new mart or a
    renamed column is covered the day it lands instead of the day someone remembers to
    update a constant.
    """
    from app.ai import warehouse  # local import: warehouse imports this module's callers
    try:
        rows = warehouse.run_sql(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "ORDER BY table_name, ordinal_position", row_cap=4000
        )["rows"]
    except Exception:
        return ()
    out = []
    for r in rows:
        col, typ = str(r.get("column_name") or ""), str(r.get("data_type") or "").upper()
        if "CHAR" not in typ and "STRING" not in typ and "VARCHAR" not in typ:
            continue
        table = str(r.get("table_name") or "")
        if table.startswith("_"):
            continue          # `_pydf_*` are internal scratch copies, not answerable sources
        if _ENTITY_COL_RE.search(col):
            out.append((table, col))
    return tuple(out)


def sql_literals(sql: str) -> list[str]:
    """The string literals a query filters on, minus the ones that mean nothing."""
    out = []
    for m in _STR_LIT_RE.finditer(sql or ""):
        v = m.group(1).replace("''", "'").strip()
        if len(v) >= 2 and v.lower() not in _NOISE and not re.fullmatch(r"[\d\-/: ]+", v):
            out.append(v)
    return out


def missing_entity_scope(sql: str, entities: list[str]) -> str | None:
    """A query run after an entity was resolved must actually mention that entity.

    `entities` are the identifying tokens of whatever the turn resolved — a material
    code, a distinctive word from its description. The check is deliberately generous:
    ANY one of them appearing anywhere in the SQL satisfies it. It exists to catch the
    filter being dropped entirely, not to police how the filter is written.
    """
    if not entities:
        return None
    hay = _norm(sql)
    if not hay:
        return None
    for e in entities:
        token = _norm(e)
        if token and len(token) >= 3 and token in hay:
            return None
    shown = ", ".join(sorted({e for e in entities if e})[:4])
    return (
        f"This query is not scoped to the item the question is about ({shown}) — none of its "
        f"identifiers appear anywhere in the SQL, so it would return figures for EVERY item "
        f"and report them as that one item's. Either filter to it, or, if this table has no "
        f"column that identifies it, say plainly that the data is not available at that level."
    )


def explain_zero_rows(sql: str, max_probe: int = 220) -> str | None:
    """A filtered query came back empty. Is the value simply somewhere else?

    "No rows" and "no such data" are different claims, and the model reports the first
    as the second. Before that becomes an answer, look the filtered literal up across
    every entity column in the warehouse. If it lives in a column the query did not
    touch, the honest finding is that the query asked the wrong dimension.
    """
    from app.ai import warehouse
    lits = sql_literals(sql)
    if not lits:
        return None
    referenced = _norm(sql)
    cols = entity_columns()
    if not cols:
        return None

    for lit in lits[:3]:
        found: list[tuple[str, str, int]] = []
        probed = 0
        for table, col in cols:
            if probed >= max_probe:
                break
            # skip the columns this query already filtered on — we know they came back empty
            if _norm(f"{table}.{col}") in referenced or _norm(col) in referenced:
                continue
            probed += 1
            try:
                r = warehouse.run_sql(
                    f"SELECT COUNT(*) AS n FROM {table} WHERE upper(CAST({col} AS VARCHAR)) "
                    f"LIKE '%' || upper('{lit.replace(chr(39), chr(39) * 2)}') || '%'",
                    row_cap=1, timeout_s=3.0,
                )
                n = int((r["rows"] or [{}])[0].get("n") or 0)
            except Exception:
                continue
            if n:
                found.append((table, col, n))
        if found:
            # Most rows first: the column that actually holds this entity dominates the
            # incidental mentions (a manufacturer's name showing up inside a product
            # description, say), and naming that column first is the whole point.
            found.sort(key=lambda x: -x[2])
            where = "; ".join(f"{t}.{c} ({n:,} rows)" for t, c, n in found[:3])
            return (
                f"0 rows — but '{lit}' DOES exist in this warehouse, under a different column: "
                f"{where}. This is a wrong-dimension query, not missing data. Re-run it against "
                f"the column that actually holds this value, and do NOT tell the user the data "
                f"does not exist."
            )
    return None


# ── CONSUMPTION IS TWO THINGS, AND THE TABLE ONLY HOLDS ONE ─────────────────────────────
# `fact_consumption` records materials ISSUED FROM STORES. It does not record materials
# dispensed against a patient's bill, and for 13,928 of the 25,166 materials in the
# catalogue — 55% of them, carrying ₹450.91 Cr of revenue — that is the ONLY way they
# move. Those items have exactly zero rows here, permanently, by construction.
#
# Asked "how does the consumption trend of Keytruda look", the assistant queried this
# table, got nothing, and reported "there is no recorded internal consumption trend" —
# for a drug with 2,193 units billed and ₹47.48 Cr of revenue. The query was right, the
# table was wrong, and nothing in the schema said so. warehouse.py has documented this
# since the billable/non-billable work; the note just never reached the SQL surface.
_CONSUMPTION_INTERNAL_ONLY = "fact_consumption"
_CONSUMPTION_BILLED = "kpi_billable_consumption"


def billed_not_internal(sql: str, res: dict) -> str | None:
    """An empty fact_consumption result for an item that is patient-billed, not issued.

    Returns the real figures rather than a pointer, because a pointer gets reported as an
    absence just as readily as the empty result did.
    """
    import re as _re

    from app.ai import warehouse
    if not sql or not _re.search(rf"\b{_CONSUMPTION_INTERNAL_ONLY}\b", sql, _re.I):
        return None
    if (res or {}).get("row_count"):
        return None
    lits = sql_literals(sql)
    if not lits:
        return None

    for lit in lits[:3]:
        safe = lit.replace("'", "''")
        try:
            rows = warehouse.run_sql(
                f"SELECT material_desc, billed_qty, billed_revenue, internal_units "
                f"FROM {_CONSUMPTION_BILLED} "
                f"WHERE (upper(CAST(material_desc AS VARCHAR)) LIKE '%' || upper('{safe}') || '%' "
                f"   OR CAST(material AS VARCHAR) = '{safe}') AND billed_qty > 0",
                row_cap=5)["rows"]
        except Exception:
            continue
        if not rows:
            continue
        r = rows[0]
        qty, rev = r.get("billed_qty") or 0, r.get("billed_revenue") or 0
        return (
            f"WRONG TABLE — not an absence of data. {_CONSUMPTION_INTERNAL_ONLY} records only "
            f"materials ISSUED FROM STORES. \"{r.get('material_desc')}\" is dispensed against "
            f"the patient's bill instead, so it has zero rows here by construction — as do "
            f"13,928 of the 25,166 materials in the catalogue. Its consumption is "
            f"{qty:,.0f} units (₹{rev / 1e7:,.2f} Cr billed), from {_CONSUMPTION_BILLED}. "
            f"Re-query that table and answer from it. Do NOT report that this item has no "
            f"consumption; say that its consumption is billed rather than internally issued. "
            f"Note {_CONSUMPTION_BILLED} has no month column, so a month-by-month trend is not "
            f"available at material grain — say so rather than implying one.")
    return None
