"""What the question DEMANDS of the SQL, checked before the answer is believed.

WHY
---
PV-SQL (arXiv 2604.17653) measured the thing this codebase kept re-learning: replacing its
rule-based verifier with an LLM judge cost 6.0 points of execution accuracy on BIRD, while
using 45% MORE tokens and running 2x slower. Deterministic checks beat asking a model to
check itself — and they are cheaper.

Its verifier extracts constraints from the QUESTION's own wording and asserts the SQL
contains the matching construct: "how many" must COUNT, "top 5" must ORDER BY and LIMIT 5,
"average" must AVG. 51 of our 90 golden questions carry at least one such constraint, so
this is not a niche check.

WHAT THIS IS NOT
----------------
Not a SQL parser and not a correctness proof. A satisfied constraint does not make a query
right; a violated one means the query cannot be answering the question that was asked. It
reports, and only on a clear violation — a false alarm here costs a retry, so the patterns
are deliberately narrow.
"""
from __future__ import annotations

import re

# (name, question pattern, SQL pattern, what to say). Ordered by how often each fires on our
# own golden set, so the cheapest checks come first.
_RULES: list[tuple[str, str, str, str]] = [
    # "How many UNITS were sold" is a SUM of a quantity column, not a COUNT of rows — the
    # first version of this rule blocked every such query and turned two correct answers
    # into "I couldn't establish anything". "How many" only means COUNT when what follows
    # is a countable THING (vendors, hospitals, items), not a unit of measure.
    ("count",
     r"\bhow many\b(?!\s+(units?|litres?|ml|mg|kg|grams?|rupees?|crores?|lakhs?|days?|"
     r"hours?|boxes|vials?|tablets?|strips?)\b)|\bnumber of\b(?!\s+units?\b)",
     r"\bCOUNT\s*\(",
     "The question asks HOW MANY, so the query must COUNT. Returning a list, or a SUM of a "
     "value column, answers a different question."),
    # A column that IS an average satisfies this without calling AVG(). kpi_vendor_lead_time
    # stores `avg_lead_time_days` pre-computed, and demanding the function blocked five
    # correct queries in a row until the engine reported "there is no lead time data for
    # Vardhman" — for a vendor whose figure is 4.77 days over 128,357 rows.
    ("average",
     r"\baverage\b|\bmean\b(?!\s*while)|\bavg\b",
     r"\bAVG\s*\(|\bMEDIAN\s*\(|\bquantile|avg_|_avg\b|average|median|mean_",
     "The question asks for an AVERAGE. A SUM or a raw list is not one — and if you mean to "
     "average a per-entity figure, say that is what you did."),
    ("percent",
     r"\bpercent|\bpercentage\b|\bratio\b|\bproportion\b|\bshare of\b",
     r"/|\bratio\b|100\s*\*|\*\s*100",
     "The question asks for a PERCENTAGE or SHARE, which requires a division. An absolute "
     "figure cannot answer it."),
    ("distinct",
     r"\bunique\b|\bdistinct\b|\bhow many (different|separate)\b",
     r"\bDISTINCT\b|\bGROUP\s+BY\b",
     "The question asks for UNIQUE values, so the query needs DISTINCT (or a GROUP BY). "
     "Counting rows counts duplicates."),
    ("topk",
     r"\btop\s+(\d+|five|ten|three|four)\b|\bfirst\s+(\d+|five|ten|three)\b",
     r"\bORDER\s+BY\b",
     "The question asks for a TOP-N, which needs an ORDER BY to define what 'top' means. "
     "Without it the rows returned are arbitrary."),
    ("extreme",
     r"\b(biggest|largest|smallest|highest|lowest|worst|best|most|least|maximum|minimum)\b",
     r"\bORDER\s+BY\b|\bMAX\s*\(|\bMIN\s*\(",
     "The question asks for an EXTREME (biggest/worst/most), which needs an ORDER BY or a "
     "MAX/MIN. Otherwise nothing establishes that the row returned is the extreme one."),
    ("temporal",
     r"\btrend\b|\bover time\b|\bmonth[- ]on[- ]month\b|\bby month\b|\bmonthly\b",
     r"\bmonth\b|\bposting_date\b|\bdate_trunc\b|\bperiod\b|\byear\b",
     "The question asks for a TREND, so the query must group by a time column. A single "
     "total has no direction and cannot show a trend."),
]

_COMPILED = [(name, re.compile(qp, re.I), re.compile(sp, re.I), msg)
             for name, qp, sp, msg in _RULES]


def required(question: str) -> list[str]:
    """Which constraints this question imposes. Useful for tests and for reporting."""
    return [n for n, qp, _, _ in _COMPILED if qp.search(question or "")]


_TABLES_IN = re.compile(r"\b(?:FROM|JOIN)\s+\"?([A-Za-z_]\w*)\"?", re.I)
_TIME_COL = re.compile(r"^(month|month_num|month_name|posting_date|date|period|year|"
                       r"snapshot_date|doc_date|week)$", re.I)


def _no_time_column_available(sql: str) -> bool:
    """True when NONE of the tables in this query carry a time column.

    Demanding a construct the data cannot supply is a deadlock, and it was one: asked for
    Keytruda's consumption trend, a query WITH `month` failed (consumption_all has no such
    column) and a query WITHOUT it was blocked by the TEMPORAL rule. Three retries, then
    "I ran into a repeated error building the query". A rule may insist the SQL match the
    question; it may not insist on a column that does not exist.
    """
    try:
        from app.ai import warehouse
        tables = set(_TABLES_IN.findall(sql or ""))
        if not tables:
            return False
        rows = warehouse.con().execute(
            "SELECT table_name, column_name FROM information_schema.columns").fetchall()
    except Exception:
        return False
    return not any(t in tables and _TIME_COL.match(c) for t, c in rows)


def _grain_is_impossible(question: str) -> bool:
    """True when the warehouse simply cannot serve this question's time grain.

    Demanding a construct the data cannot provide creates a deadlock, and it did: asked for
    Keytruda's consumption trend, a query WITH `month` failed (consumption_all has no such
    column) and a query WITHOUT it was blocked by the TEMPORAL rule. The chat retried three
    times and gave up. A constraint may insist the SQL match the question; it may not insist
    on a column that does not exist.
    """
    try:
        from app.ai import resolve
        return resolve.impossible_combination(question) is not None
    except Exception:
        return False


def violations(question: str, sql: str) -> list[str]:
    """Constraints the question imposes that the SQL does not satisfy."""
    if not question or not sql:
        return []
    skip = set()
    if _grain_is_impossible(question) or _no_time_column_available(sql):
        skip.add("temporal")
    # a CTE or subquery may satisfy the construct anywhere in the statement, so the whole
    # text is searched rather than just the outer SELECT
    return [f"{name.upper()} — {msg}"
            for name, qp, sp, msg in _COMPILED
            if name not in skip and qp.search(question) and not sp.search(sql)]


# A month key that is not monotonic across years. ORDER BY month_num runs
# January(1) … December(12), so on data spanning Dec 2025 -> May 2026 December lands LAST
# and the narrative names the wrong peak, trough and direction. Every such table now carries
# a `period` column ('2025-12') that sorts correctly.
_BAD_TIME_SORT = re.compile(
    r"\bORDER\s+BY\b[^;]*?\b(month_num|month)\b", re.I)
_HAS_PERIOD = re.compile(r"\bperiod\b", re.I)
_HAS_YEAR_FIRST = re.compile(r"\bORDER\s+BY\s+[^;]{0,40}?\byear\b", re.I)


# A month DERIVED from a real date sorts correctly, whatever it is aliased as:
# `DATE_TRUNC('month', CAST(posting_date AS DATE)) AS month ... ORDER BY month` is right,
# and the first version of this rule blocked it on sight of the word "month" — leaving the
# engine with no usable query at all and the answer "no conclusions can be drawn".
_DERIVED_MONTH = re.compile(
    r"date_trunc|strftime|date_part|extract\s*\(|::\s*date|AS\s+DATE\s*\)", re.I)


def wrong_time_ordering(sql: str) -> str | None:
    """A trend ordered by a key that puts December before January."""
    if not sql or not _BAD_TIME_SORT.search(sql):
        return None
    if _HAS_PERIOD.search(sql) or _HAS_YEAR_FIRST.search(sql):
        return None                      # already sorted by something monotonic
    if _DERIVED_MONTH.search(sql):
        return None                      # month came from a real date; it sorts fine
    return ("WRONG TIME ORDER — `month_num` is the calendar month 1-12 with the year in a "
            "separate column, and `month` may be a NAME. This data spans December 2025 to "
            "May 2026, so ordering by either puts December LAST when it is the FIRST "
            "period, and the peak, trough and direction all come out wrong. Order by the "
            "`period` column instead ('2025-12', '2026-01', …), which every month-grained "
            "table now carries.")


_NOT_SQL = re.compile(r"^\s*(--|#)|^\s*$")


def check(question: str, sql: str) -> str | None:
    """One message naming every unmet constraint, or None when the SQL satisfies them all.

    Canonical KPI results arrive here as a comment (`-- get_kpi('near-expiry')`) rather than
    a query. They are correct by construction and contain no SQL to inspect, so checking
    them produced pure false alarms — 2 of the first 6 questions tried, on the one path that
    never needed checking.
    """
    if not sql or _NOT_SQL.match(sql) or not re.search(r"\bSELECT\b", sql, re.I):
        return None
    bad = violations(question, sql)
    for extra in (wrong_time_ordering(sql), misleading_alias(sql),
                  under_aggregated_trend(question, sql)):
        if extra:
            bad = bad + [extra]
    if not bad:
        return None
    return ("QUERY DOES NOT MATCH THE QUESTION — re-write it before using the result.\n"
            + "\n".join(f"- {b}" for b in bad))


# ── AN ALIAS MAY NOT RENAME THE MEASURE ──────────────────────────────────────────────────
# `SELECT SUM(monthly_purchase_value) AS total_sales_value` — a PURCHASING column relabelled
# as SALES. The engine had already disclosed, deterministically, "what follows is
# PURCHASING", and then the prose followed the alias and called the same numbers "total
# sales value" in the very next sentence. Two stacked, contradictory claims about one figure.
#
# Disclosure cannot survive a query that lies in its own column names, and no amount of
# prompt text fixes it: the model reads its own alias back and believes it.
_FAMILY_OF_COLUMN: tuple[tuple[str, str], ...] = (
    ("purchasing",  r"purchase|procure|po_value|line_value|grn|spend"),
    ("sales",       r"revenue|sales|billed_revenue|turnover"),
    ("consumption", r"consum|issued|dispens"),
    ("stock",       r"stock|inventory|on_hand|closing"),
)


def _family_of(name: str) -> str | None:
    for fam, pat in _FAMILY_OF_COLUMN:
        if re.search(pat, name or "", re.I):
            return fam
    return None


_TREND_COL = re.compile(r"\b(period|month|month_num|posting_date)\b", re.I)
_AGGREGATED = re.compile(r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(|\bGROUP\s+BY\b", re.I)


def under_aggregated_trend(question: str, sql: str) -> str | None:
    """A per-period series that never groups, so a period can appear more than once.

    `SELECT period, revenue FROM sales_by_material_month WHERE material_desc = '…'` returns
    TWO rows for December when two material codes share that description, and the answer
    read "December 2025: ₹9.29 Cr + ₹38.97 L" — a sum the reader is left to do, in mixed
    units. A trend has exactly one value per period or it is not a trend.
    """
    if not sql or not question:
        return None
    if not re.search(r"\btrend\b|\bover time\b|\bmonth(ly|-on-month)?\b", question, re.I):
        return None
    if not _TREND_COL.search(sql) or _AGGREGATED.search(sql):
        return None
    return ("UNDER-AGGREGATED TREND — this selects a measure per period without SUM and "
            "without GROUP BY, so any period with more than one matching row appears twice "
            "and the reader is handed two numbers to add. Group by the period column and "
            "SUM the measure, so each period has exactly one value.")


def misleading_alias(sql: str) -> str | None:
    """An aggregate aliased into a different measure family than its source column."""
    if not sql:
        return None
    for raw_col, alias in re.findall(
            r"\b(?:SUM|AVG|MIN|MAX|COUNT)\s*\(\s*(?:DISTINCT\s+)?([A-Za-z_][\w.]*)[^()]*\)"
            r"\s*(?:AS\s+)?([A-Za-z_]\w*)", sql, re.I):
        src = _family_of(raw_col.split(".")[-1])
        dst = _family_of(alias)
        if src and dst and src != dst:
            return (f"MISLEADING ALIAS — `{raw_col}` is a {src.upper()} column and you have "
                    f"named it `{alias}`, which reads as {dst.upper()}. Those are different "
                    f"business events measured from different tables, and the answer will "
                    f"call the figure by the alias you chose. Name it after what it is "
                    f"(e.g. total_{src}_value) and describe it as {src.upper()}.")
    return None
