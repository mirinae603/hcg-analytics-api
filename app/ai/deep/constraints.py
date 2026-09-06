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
    ("count",
     r"\bhow many\b|\bnumber of\b",
     r"\bCOUNT\s*\(",
     "The question asks HOW MANY, so the query must COUNT. Returning a list, or a SUM of a "
     "value column, answers a different question."),
    ("average",
     r"\baverage\b|\bmean\b(?!\s*while)|\bavg\b",
     r"\bAVG\s*\(|\bMEDIAN\s*\(|\bquantile",
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


def violations(question: str, sql: str) -> list[str]:
    """Constraints the question imposes that the SQL does not satisfy."""
    if not question or not sql:
        return []
    # a CTE or subquery may satisfy the construct anywhere in the statement, so the whole
    # text is searched rather than just the outer SELECT
    return [f"{name.upper()} — {msg}"
            for name, qp, sp, msg in _COMPILED
            if qp.search(question) and not sp.search(sql)]


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
    if not bad:
        return None
    return ("QUERY DOES NOT MATCH THE QUESTION — re-write it before using the result.\n"
            + "\n".join(f"- {b}" for b in bad))
