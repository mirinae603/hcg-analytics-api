# tests/test_answer_gate.py — checks on the ANSWER, not on the query behind it.
#
# THE STRUCTURAL PROBLEM THIS SOLVES. All eight existing guards live in run_query / run_sql:
# part_exceeds_whole, missing_entity_scope, constraints.check, coverage.disclosure,
# billed_not_internal, placeholder_won_a_ranking, sink_placeholders, _kpi_scope_mismatch.
# Every one of them inspects SQL, so every one of them is bypassed by a route that produces
# none — the canonical KPI path (twice), and the give-up path, which runs no query at all.
#
# Each fix I made covered one more route, and the next new route bypassed it again. These
# checks read the finished text, which every answer passes through however it was built.
from __future__ import annotations

from app.ai.deep.answer_gate import (check, gave_up_without_looking,
                                     substitution_without_alternative, unnamed_bucket_leads)


def test_an_unnamed_bucket_as_the_headline_is_caught_whatever_route_made_it():
    # sink_placeholders removes these from RANKED ROWS; the same value reached the prose by
    # another route and led the answer anyway
    assert unnamed_bucket_leads(
        "The largest spend category is **Uncategorized**, with ₹173.31 Cr.")
    assert unnamed_bucket_leads(
        "The biggest category was: Unclassified at ₹12 Cr.")


def test_a_named_leader_passes():
    assert unnamed_bucket_leads(
        "The largest spend category is ANTINEOPLASTIC at ₹84.45 Cr.") is None


def test_a_bare_substitution_is_flagged():
    # honest, and unhelpful: it swaps the measure and never says why or what does exist
    assert substitution_without_alternative(
        "There is no sales figure at this level; what follows is STOCK. Stock is ₹60 Cr.")


def test_a_substitution_that_explains_itself_passes():
    assert substitution_without_alternative(
        "There is no sales figure; what follows is STOCK, because sales and plant codes "
        "share no rows. Sales IS available by hospital.") is None


def test_giving_up_after_zero_queries_is_flagged():
    # an unanswerable question and an unattempted one produce the same sentence
    assert gave_up_without_looking("I couldn't establish anything solid enough to report.", 0)


def test_giving_up_after_really_looking_is_not():
    assert gave_up_without_looking(
        "I couldn't establish anything solid enough to report.", 4) is None


def test_a_good_answer_trips_nothing():
    assert check("Monthly revenue fell from ₹10.91 Cr in December to ₹7.87 Cr in May.", 3) == []


def test_the_gate_never_raises_on_junk():
    assert check("", 0) == [] and check(None, 0) == []


# ── the precondition that stops the give-up in the first place ──────────────────────────
def test_give_up_requires_evidence_of_looking():
    # the tool description said "call this only after LOOKING" — advice, which lost. Asked
    # whether Reliance is a supplier or a manufacturer, the worker gave up having run ZERO
    # queries, for a fact one lookup away.
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "ai" / "deep" / "engine.py").read_text()
    blk = src[src.index('if name == "give_up":'):][:2000]
    # the message tightened once: describing a table used to count as looking, and a worker
    # that read a schema and concluded from it still gave up having run ZERO queries
    assert "You have not RUN anything" in blk
    assert "run_query" in blk and "seen_calls" in blk


def test_the_engine_retries_a_run_that_flagged_itself():
    """The mechanism that attacks route variance directly.

    Three runs of the whole bank found NOT ONE case failing all three times — every question
    is answered correctly sometimes, so the residual error is variance between runs. That is
    only fixable if a bad run can be RECOGNISED, and it can: the failing run in a six-case
    probe came back `flagged`.

    Flagged fires on good runs too, and for a retry that asymmetry is the right way round —
    a false alarm costs a second attempt, a missed one costs a wrong answer.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "ai" / "deep" / "engine.py").read_text()
    assert "def _answer_once(" in src
    assert "retry_allowed" in src
    blk = src[src.index("if retry_allowed and verified"):][:600]
    assert 'saw_answer.get("verified") != "flagged"' in blk, "must prefer the UNflagged run"


def test_the_retry_cannot_recurse():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "app" / "ai" / "deep" / "engine.py").read_text()
    assert "_answer_once(query, history, retry_allowed=False)" in src
