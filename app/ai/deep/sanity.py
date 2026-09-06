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


def sink_placeholders(res: dict) -> bool:
    """Move unnamed buckets to the BOTTOM of a result. True if anything moved.

    placeholder_leader() warns, and the warning was read past: "our biggest spend category
    is Uncategorized, ₹173.31 Cr" appeared with the caution sitting right beside it. Every
    consumer of a result — the ranking shape, the chart, the writer skimming row one —
    treats position as rank, so the fix belongs in the rows.

    The bucket is not dropped. It is real, and its size is a finding about data quality;
    it just stops being the answer to "which category is biggest".
    """
    rows = (res or {}).get("rows") or []
    if len(rows) < 2:
        return False
    label_col = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
    if not label_col:
        return False
    named = [r for r in rows if not _PLACEHOLDER.match(str(r.get(label_col) or ""))]
    unnamed = [r for r in rows if _PLACEHOLDER.match(str(r.get(label_col) or ""))]
    if not unnamed or not named:
        return False
    res["rows"] = named + unnamed
    return True


def city_on_unreachable_table(sql: str) -> str | None:
    """Block a city filter on a table whose site codes cannot reach dim_plant.

    The brief says plainly that sales_by_hospital.hospital shares ZERO codes with dim_plant
    and that no sales figure can be scoped to a city. "KEYTRUDA, ₹10.78 Cr in Bangalore
    hospitals" was written anyway, three runs apart, because a statement in a prompt is a
    preference and the query still ran. This makes it a precondition: the query fails and
    the model is told why, in the one place it cannot skim past.
    """
    if not sql:
        return None
    from app.ai import resolve as _resolve
    _, blocked = _resolve.city_reachability()
    tables = {t.split(".")[0] for t in blocked}
    hit = next((t for t in tables if re.search(rf"\b{re.escape(t)}\b", sql, re.I)), None)
    if not hit:
        return None
    city = next((c for c in _resolve.cities()
                 if re.search(rf"\b{re.escape(c)}\b", sql, re.I)), None)
    if not city:
        return None
    return (f"IMPOSSIBLE FILTER — {hit} cannot be scoped to a city. Its site codes share no "
            f"value with dim_plant, where city names live, so there is no join and no filter "
            f"that yields {city}'s sales. Any number this query returns is some other scope "
            f"wearing {city}'s label. Answer at the level the data supports — company-wide, "
            f"or per hospital using that table's own site codes — and say why the city cut "
            f"is not available.")


def placeholder_won_a_ranking(sql: str, res: dict, question: str = "") -> str | None:
    """Refuse a ranking whose winner is an unnamed bucket.

    sink_placeholders fixes a result that CONTAINS the named categories. It cannot fix
    `ORDER BY spend DESC LIMIT 1`, which returns one row reading "Uncategorized, ₹173.31
    Cr" — there is nothing left to promote. The query itself is the wrong query: nobody
    asking for their biggest spend category is asking how much of the spend is unclassified.

    Left alone when the question really is about the gap, because then it is the answer.
    """
    if not sql or not re.search(r"\bORDER\s+BY\b", sql, re.I):
        return None
    if re.search(r"uncategori[sz]ed|unclassified|unknown|missing|blank|null|not assigned",
                 question or "", re.I):
        return None                       # they asked about the gap; that is legitimate
    rows = (res or {}).get("rows") or []
    if not rows:
        return None
    label_col = next((k for k, v in rows[0].items() if isinstance(v, str)), None)
    if not label_col or not _PLACEHOLDER.match(str(rows[0].get(label_col) or "")):
        return None
    named = next((r for r in rows[1:]
                  if not _PLACEHOLDER.match(str(r.get(label_col) or ""))), None)
    if named:
        return None                       # a real category is present; ordering handles it
    return (f"WRONG QUERY — the top row is \"{rows[0][label_col]}\", which is an absence of "
            f"data, not a category. Re-run this with the unnamed buckets excluded — add "
            f"WHERE {label_col} IS NOT NULL AND trim({label_col}) <> '' AND upper({label_col}) "
            f"NOT IN ('UNCATEGORIZED','UNCLASSIFIED','UNKNOWN','OTHER','N/A','NONE') — so the "
            f"ranking is of things that have names. Report the unclassified share separately "
            f"as a data-quality point; it is worth saying, but it is not the answer.")
