# tests/test_grain_availability.py — telling the user what the data cannot do, by name.
#
# "Show me the sales trend for KEYTRUDA 100MG INJ VIAL" ruled out both lines of enquiry and
# answered "I couldn't establish anything solid enough to report". That reads as the
# assistant being weak. The truth is that `sales_monthly` has month and revenue but no
# material, `sales_by_material` has material and revenue but no month, and no table has all
# three — the question is impossible, not hard, and saying so is the correct answer.
from __future__ import annotations

from app.ai.resolve import brief, grain_measure_tables, impossible_combination


def test_a_monthly_trend_for_one_drug_is_named_impossible():
    r = impossible_combination('Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"')
    assert r and "MONTH x MATERIAL" in r
    assert "does not exist in this warehouse" in r


def test_it_offers_what_does_exist_instead():
    r = impossible_combination('Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"')
    assert "IS available by" in r
    assert "do not report it as a failure" in r.lower() or "not report it as a failure" in r


def test_an_answerable_question_is_left_alone():
    assert impossible_combination("How has monthly revenue moved over the period?") is None
    assert impossible_combination("what is revenue by hospital") is None


def test_projections_do_not_count_as_history():
    # forecast_sales carries material, month AND a sales value, so a naive search concludes
    # a per-drug monthly sales trend exists — out of forecast rows
    assert not any("forecast" in t for t in grain_measure_tables("revenue", "month"))
    assert any("forecast" in t for t in grain_measure_tables("revenue", "month", True))


def test_but_a_forecast_question_may_use_them():
    assert impossible_combination(
        "show me the forecast sales trend for KEYTRUDA 100MG INJ VIAL") is None


def test_the_brief_says_it_before_any_query_runs():
    b = brief('Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"')
    assert "NOT AVAILABLE AT THIS GRAIN" in b


def test_a_question_with_no_measure_or_grain_is_not_judged():
    assert impossible_combination("hello") is None
