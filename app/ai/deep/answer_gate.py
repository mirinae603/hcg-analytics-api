"""Checks on the ANSWER, not on the query that produced it.

WHY THIS EXISTS
---------------
Every other guard in this system inspects SQL — part_exceeds_whole, missing_entity_scope,
constraints.check, coverage.disclosure, billed_not_internal, placeholder_won_a_ranking. All
of them live in `run_query` / `run_sql`, and that is the whole problem:

  - the canonical KPI path produces NO SQL, so every one of them sits it out. Twice now a
    figure reached the user through it that a query would have been blocked for.
  - the give-up path runs no query AT ALL, so there is nothing for a query-guard to see.
  - a fix to one guard covers one more route; the next new route bypasses it again.

Answers leave the engine at two points regardless of how they were built. Checking there
means a check cannot be dodged by taking a different path to the same sentence. This module
holds only checks that can be made from the finished text plus what the engine already
knows — no re-querying, no second model.

It REPORTS. It does not rewrite: the correction depends on what the reader asked, and
guessing at it trades a visibly poor answer for an invisibly wrong one.
"""
from __future__ import annotations

import re

# An unnamed bucket presented as the answer. `sink_placeholders` removes these from ranked
# rows, but only on results that pass through a ranking derivation — the same value can
# reach the prose by another route, and did.
_UNNAMED = r"(uncategori[sz]ed|unclassified|unknown|not assigned|unspecified|n/?a|others?)"
_LEADS_WITH_UNNAMED = re.compile(
    r"\b(largest|biggest|top|highest|leading|most)\b[^.]{0,60}?\b(is|was|:)\s*\**\s*['\"]?"
    + _UNNAMED, re.I)

# "what follows is STOCK" — an honest substitution, which becomes unhelpful when the answer
# never says what the thing actually asked for IS available by.
_SUBSTITUTION = re.compile(r"what follows is\s+([A-Z][A-Z_ ]{2,30})", re.I)
_OFFERS_ALTERNATIVE = re.compile(
    r"\bis available\b|\bavailable by\b|\byou can\b|\bwould you like\b|\binstead\b|"
    r"\bclosest\b|\bcannot be joined\b|\bshares? no\b|\bno column\b", re.I)


def unnamed_bucket_leads(prose: str) -> str | None:
    """The answer names an absence of data as its headline."""
    if not prose:
        return None
    m = _LEADS_WITH_UNNAMED.search(prose)
    if not m:
        return None
    return (f"The answer leads with \"{m.group(3)}\" — a bucket with no name is the absence "
            f"of a classification, not the answer to which one is biggest. Report the "
            f"largest NAMED one and give the unclassified share as a separate data-quality "
            f"note.")


def substitution_without_alternative(prose: str) -> str | None:
    """A measure was swapped and the reader was never told where the real one lives."""
    if not prose:
        return None
    m = _SUBSTITUTION.search(prose)
    if not m or _OFFERS_ALTERNATIVE.search(prose):
        return None
    return (f"This answer substitutes {m.group(1).strip()} for what was asked and stops "
            f"there. Say WHY the asked-for measure cannot be given at this grain, and name "
            f"the closest thing that does exist — a substitution with no explanation and no "
            f"alternative reads as an evasion.")


def gave_up_without_looking(prose: str, query_count: int) -> str | None:
    """A limit of the data claimed without evidence of having checked."""
    if not prose or query_count > 0:
        return None
    if not re.search(r"couldn't establish|not answerable|isn't answerable|no data", prose, re.I):
        return None
    return ("This reports a limit of the DATA after running ZERO queries, which is a limit "
            "of the ATTEMPT. An unanswerable question and an unattempted one produce the "
            "same sentence and only one of them is honest.")


def check(prose: str, query_count: int = 0) -> list[str]:
    """Every problem visible in the finished answer."""
    return [w for w in (unnamed_bucket_leads(prose),
                        substitution_without_alternative(prose),
                        gave_up_without_looking(prose, query_count)) if w]
