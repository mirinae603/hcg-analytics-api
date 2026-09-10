"""A part cannot be larger than the whole.

The most damaging error this engine makes is not a missing answer, it is a confident inflated
one. "Our procurement spend is dominated by Consumables, ₹19,825.46 Cr" reads perfectly and is
about forty times the entire procurement spend in the warehouse (₹478.27 Cr). Nothing caught
it: the SQL was well-formed, it was scoped to the right entity, it violated no ontology rule,
and the corroborator agreed because both derivations shared the same inflated join.

The invariant is arithmetic rather than semantic, which is what makes it worth having. For a
column that never goes negative, SUM over any filter and any grouping is bounded by SUM over
the whole table. Exceed that and the rows were multiplied by a join — there is no reading of
the data under which the answer is right.

It fires on the class, not on an instance. Any fan-out, in any table, through any join, on any
measure, is caught by the same comparison, and a new table is covered the day it is added.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Floating point, and measures stored at different precisions across tables. A real fan-out
# duplicates rows, so it overshoots by a multiple, not by a fraction of a percent.
SLACK = 1.01

_SUM = re.compile(r"\bSUM\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)\s*\)"
                  r"(?:\s+AS\s+([A-Za-z_]\w*))?", re.I)


@lru_cache(maxsize=512)
def _whole(table: str, column: str) -> float | None:
    """The largest value SUM(column) can legitimately take over any subset of this table.

    Not SUM(column). Measures here carry credits and returns, so the net total is SMALLER
    than what a positive-only subset can reach, and comparing against it would reject correct
    answers. The sum of the POSITIVE values is a true upper bound for any filter and any
    grouping, and a fan-out multiplies those positives, so it still overshoots.

    The first version of this bailed out whenever the column had a negative value, which
    disabled the check on line_value — the exact column whose fan-out shipped a figure forty
    times the size of the warehouse.
    """
    from app.ai import warehouse
    try:
        r = warehouse.run_sql(
            f'SELECT SUM("{column}") FILTER (WHERE "{column}" > 0) AS cap FROM "{table}"',
            row_cap=1)
        cap = (r.get("rows") or [{}])[0].get("cap")
        return float(cap) if cap is not None else None
    except Exception:
        return None


def _owner(column: str, tables: list[str]) -> str | None:
    """Which table in this query owns the column, according to the ontology."""
    from app.ai import ontology
    cols = ontology.load().get("columns") or {}
    hits = [t for t in tables if f"{t}.{column}" in cols]
    # Ambiguous ownership means the bound could be taken from the wrong table, and a wrong
    # bound would reject correct answers. Decline instead.
    return hits[0] if len(hits) == 1 else None


def _tables(sql: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b(?:FROM|JOIN)\s+[\"']?([A-Za-z_]\w*)[\"']?", sql, re.I)))


def exceeds_the_whole(sql: str, result: dict) -> str | None:
    """The reported total against the table's own total for the same measure."""
    if not result or result.get("truncated"):
        # A truncated result sums to less than it should, so the comparison can only produce
        # a false negative — but say so rather than pretending the check ran.
        return None
    rows = result.get("rows") or []
    if not rows:
        return None
    tables = _tables(sql)
    if not tables:
        return None

    for inner, alias in _SUM.findall(sql):
        column = inner.split(".")[-1]
        owner = _owner(column, tables)
        if not owner:
            continue
        whole = _whole(owner, column)
        if not whole or whole <= 0:
            continue
        key = alias or next((k for k in rows[0]
                             if k.lower() in (f"sum({inner})".lower(), column.lower())), None)
        if not key or key not in rows[0]:
            continue
        try:
            part = sum(float(r[key]) for r in rows if r.get(key) is not None)
        except (TypeError, ValueError):
            continue
        if part > whole * SLACK:
            return (
                f"That query returns {part:,.0f} for SUM({column}), but the whole of "
                f"{owner}.{column} is only {whole:,.0f} — the result is {part / whole:.1f}x the "
                f"entire table. A join has multiplied the rows before the sum. Aggregate "
                f"{owner} on its own first and join the result, or join on a key that is "
                f"unique on the other side.")
    return None
