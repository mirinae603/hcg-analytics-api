# tests/test_constraints.py — what the QUESTION demands of the SQL.
#
# PV-SQL (arXiv 2604.17653) measured rule-based verification beating an LLM judge by 6.0
# points of execution accuracy at 45% fewer tokens. Its verifier asserts the SQL contains
# the construct the question's wording requires.
from __future__ import annotations

import pytest

from app.ai.deep.constraints import check, required


def test_a_share_question_must_divide():
    # "What share of revenue is non-formulary" was answered "₹29.06 Cr" — the right
    # numerator, never divided, so no share was ever given
    assert check("What share of our sales revenue is non-formulary?",
                 "SELECT SUM(revenue) FROM sales_by_material WHERE x") is not None
    assert check("What share of our sales revenue is non-formulary?",
                 "SELECT SUM(a)/SUM(b)*100 FROM t") is None


@pytest.mark.parametrize("q", [
    "How many units of KEYTRUDA were sold?",
    "How many units are expiring in the next 90 days?",
    "How many vials did we consume?",
    "How many days of cover do we hold?",
])
def test_how_many_UNITS_is_a_sum_not_a_count(q):
    # the first version of this rule demanded COUNT(*) for these and turned two correct
    # answers into "I couldn't establish anything"
    assert "count" not in required(q), q
    assert check(q, "SELECT SUM(qty) FROM t") is None


@pytest.mark.parametrize("q", [
    "How many vendors do we buy from?",
    "How many hospitals do we operate?",
    "What is the number of materials in the catalogue?",
])
def test_how_many_THINGS_still_demands_a_count(q):
    assert "count" in required(q), q
    assert check(q, "SELECT material FROM t GROUP BY 1") is not None
    assert check(q, "SELECT COUNT(DISTINCT material) FROM t") is None


def test_an_extreme_needs_an_ordering():
    assert check("What is our biggest spend category?",
                 "SELECT category, SUM(v) FROM t GROUP BY 1") is not None
    assert check("What is our biggest spend category?",
                 "SELECT category, SUM(v) s FROM t GROUP BY 1 ORDER BY s DESC") is None


def test_a_trend_needs_a_time_column():
    assert check("How has monthly revenue moved?", "SELECT SUM(revenue) FROM t") is not None
    assert check("How has monthly revenue moved?",
                 "SELECT month, SUM(revenue) FROM t GROUP BY 1") is None


def test_an_average_is_not_a_sum():
    assert check("What is the average lead time?", "SELECT SUM(d) FROM t") is not None
    assert check("What is the average lead time?", "SELECT AVG(d) FROM t") is None


def test_canonical_kpi_calls_are_not_sql_and_are_ignored():
    # they arrive as a comment and are correct by construction; checking them produced pure
    # false alarms on 2 of the first 6 questions tried
    assert check("How many units are expiring?", "-- get_kpi('near-expiry')") is None
    assert check("anything", "") is None
