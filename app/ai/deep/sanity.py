"""Arithmetic that must hold, checked against the warehouse rather than trusted.

WHY
---
"How much did Bangalore hospitals spend on procurement?" was answered ₹649.57 Cr. Total
procurement in the entire warehouse is ₹478.27 Cr. Four hospitals cannot spend more than
every hospital, and no amount of schema description prevents this: the model joined a fact
table to a dimension to reach the city name, the join fanned rows out, and the number came
back inflated but perfectly well-formed. Nothing downstream could tell it was wrong,
because a plausible number carries no error.

A person catches this instantly — not by re-reading the SQL, but by knowing roughly what
the total is and noticing the part is bigger. That check is cheap and mechanical, so it
should not depend on anyone remembering to do it.

WHAT THIS IS NOT
----------------
Not a SQL validator and not a linter. It runs ONE extra query — the same aggregate with the
filter removed — and compares. It only ever reports; it never rewrites the query, because
the right correction depends on intent and guessing at it would trade a loud wrong answer
for a quiet one.
"""
from __future__ import annotations

import re

from app.ai import warehouse

# SUM(x) is additive over rows, so a filter can only ever remove from it. AVG/MIN/MAX/COUNT
# DISTINCT are not, and comparing them part-to-whole proves nothing.
_SUM = re.compile(r"\bSUM\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)\s*\)", re.I)
_FROM = re.compile(r"\bFROM\s+([A-Za-z_]\w*)", re.I)
_HAS_WHERE = re.compile(r"\bWHERE\b", re.I)
_HAS_JOIN = re.compile(r"\bJOIN\b", re.I)
_TOLERANCE = 1.005          # a part may equal the whole; 0.5% absorbs float drift


def _columns(table: str) -> set[str]:
    try:
        rows = warehouse.con().execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table]).fetchall()
    except Exception:
        return set()
    return {str(r[0]).lower() for r in rows}


def part_exceeds_whole(sql: str, res: dict) -> str | None:
    """Report a subtotal larger than its own unfiltered total, or None if nothing is wrong.

    Only fires when the comparison is meaningful: an additive SUM of a plain column, over a
    single base table, narrowed by a WHERE or widened by a JOIN.
    """
    if not sql or not isinstance(res, dict):
        return None
    rows = res.get("rows") or []
    if not rows:
        return None
    if not (_HAS_WHERE.search(sql) or _HAS_JOIN.search(sql)):
        return None                       # nothing narrowed it; there is no "part"
    if re.search(r"\bUNION\b", sql, re.I):
        return None

    m, f = _SUM.search(sql), _FROM.search(sql)
    if not (m and f):
        return None
    col, table = m.group(1).split(".")[-1].lower(), f.group(1)
    if col not in _columns(table):
        return None                       # the measure is not this table's own column

    try:
        whole = warehouse.con().execute(
            f'SELECT SUM("{col}") FROM "{table}"').fetchone()[0]
    except Exception:
        return None
    if whole is None or whole <= 0:
        return None

    # the largest single number the query returned, and their sum for a breakdown
    numbers = [v for r in rows for v in r.values() if isinstance(v, (int, float))]
    part = max((abs(v) for v in numbers), default=0.0)
    if part <= whole * _TOLERANCE:
        return None
    return (f"IMPOSSIBLE RESULT — do not report this number. This query returned "
            f"{part:,.0f} for {col}, but {table} contains only {whole:,.0f} in total. A "
            f"filtered subtotal cannot exceed the unfiltered total, so the query is "
            f"over-counting — almost always a JOIN that repeats fact rows once per matching "
            f"dimension row. Aggregate the fact table on its own key, or filter it by code "
            f"instead of joining out to a name.")


_PLACEHOLDER = re.compile(
    r"^\s*(uncategori[sz]ed|unclassified|unknown|others?|n/?a|none|null|blank|not assigned|-|)\s*$",
    re.I)


def placeholder_leader(res: dict) -> str | None:
    """Warn when the top row of a breakdown is an unnamed bucket.

    "What is our biggest spend category?" was answered "Uncategorized, ₹173.31 Cr". That is
    a true row and a false answer: it names the gap in the data, not the biggest category.
    The ranking shape already demotes these, but findings that never pass through a shape
    reached the headline unchecked, so the check belongs on the raw result too.
    """
    rows = (res or {}).get("rows") or []
    if len(rows) < 2:
        return None
    label_col = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
    num_col = next((k for k, v in rows[0].items() if isinstance(v, (int, float))), None)
    if not label_col or not num_col:
        return None
    if not _PLACEHOLDER.match(str(rows[0].get(label_col) or "")):
        return None
    named = next((r for r in rows[1:]
                  if not _PLACEHOLDER.match(str(r.get(label_col) or ""))), None)
    if not named:
        return None
    return (f"The top bucket here is \"{rows[0][label_col]}\" — an absence of data, not a "
            f"category. Report \"{named[label_col]}\" as the largest NAMED "
            f"{label_col}, and mention the unclassified share separately as a data-quality "
            f"point. Never give an unnamed bucket as the answer.")


def check(sql: str, res: dict) -> list[str]:
    """Every warning that applies to one finding."""
    return [w for w in (part_exceeds_whole(sql, res), placeholder_leader(res)) if w]
