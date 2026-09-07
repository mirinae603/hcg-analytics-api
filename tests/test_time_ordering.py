# tests/test_time_ordering.py — time has to run forwards, and a column may not be renamed
# into a different measure.
#
# A live answer read: "declining from ₹8.19 Cr in January (the peak) to ₹8.01 Cr in December
# (the last period)". December 2025 is the FIRST period and May 2026 the last. `month_num`
# is the calendar month 1-12 with the year in a separate column, so ordering by it sorts
# December to the end and the narrative named the wrong peak, trough and direction — while
# the chart, built from the same rows, drew the series correctly.
from __future__ import annotations

import pytest

from app.ai.deep.constraints import misleading_alias, wrong_time_ordering


@pytest.mark.parametrize("sql", [
    "SELECT month, SUM(x) FROM mart_procurement GROUP BY 1 ORDER BY month_num",
    "SELECT month, SUM(x) FROM kpi_stock_change GROUP BY 1 ORDER BY month",
])
def test_a_non_monotonic_month_key_is_refused(sql):
    w = wrong_time_ordering(sql)
    assert w and "WRONG TIME ORDER" in w
    assert "period" in w


@pytest.mark.parametrize("sql", [
    "SELECT period, SUM(x) FROM mart_procurement GROUP BY 1 ORDER BY period",
    "SELECT year, month_num, SUM(x) FROM t GROUP BY 1,2 ORDER BY year, month_num",
    "SELECT month, SUM(x) s FROM t GROUP BY 1 ORDER BY s DESC",
    "SELECT SUM(x) FROM t",
])
def test_a_monotonic_or_irrelevant_ordering_is_allowed(sql):
    assert wrong_time_ordering(sql) is None


def test_every_month_grained_table_has_a_period_key():
    # discovered from the schema, not a hand-kept list — a table added later gets one too
    from app.ai.warehouse import con
    rows = con().execute(
        "SELECT DISTINCT table_name FROM information_schema.columns "
        "WHERE column_name = 'period'").fetchall()
    named = {r[0] for r in rows if not r[0].startswith("_period")}
    assert {"mart_procurement", "fact_consumption", "kpi_monthly_purchase_value"} <= named


def test_the_period_key_sorts_december_first():
    from app.ai.warehouse import con
    periods = [r[0] for r in con().execute(
        "SELECT DISTINCT period FROM mart_procurement ORDER BY period LIMIT 6").fetchall()]
    assert periods[0] == "2025-12"
    assert periods == sorted(periods)


# ── an alias may not rename the measure ─────────────────────────────────────────────────
def test_a_purchasing_column_may_not_be_aliased_as_sales():
    # `SUM(monthly_purchase_value) AS total_sales_value` — the engine had already disclosed
    # "what follows is PURCHASING" and the prose then called it "total sales value"
    w = misleading_alias("SELECT SUM(monthly_purchase_value) AS total_sales_value FROM t")
    assert w and "MISLEADING ALIAS" in w
    assert "PURCHASING" in w and "SALES" in w


def test_an_honest_alias_passes():
    assert misleading_alias(
        "SELECT SUM(monthly_purchase_value) AS total_purchase_value FROM t") is None
    assert misleading_alias("SELECT SUM(revenue) AS total_revenue FROM t") is None


def test_a_neutral_alias_is_not_second_guessed():
    assert misleading_alias("SELECT SUM(line_value) AS total FROM t") is None


@pytest.mark.parametrize("sql", [
    "SELECT DATE_TRUNC('month', CAST(posting_date AS DATE)) AS month, SUM(v) FROM t "
    "GROUP BY 1 ORDER BY month",
    "SELECT strftime(posting_date, '%Y-%m') AS month, SUM(v) FROM t GROUP BY 1 ORDER BY month",
    "SELECT EXTRACT(month FROM posting_date) AS month, SUM(v) FROM t GROUP BY 1 ORDER BY month",
])
def test_a_month_derived_from_a_real_date_is_allowed(sql):
    # the first version of this rule blocked these on sight of the word "month", leaving the
    # engine with no usable query at all and an answer of "no conclusions can be drawn"
    assert wrong_time_ordering(sql) is None
