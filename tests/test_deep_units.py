# tests/test_deep_units.py — unit typing in the deep engine.
#
# Caught in a live deep-mode brief: "value share percentages as high as ₹85". The column
# `value_share_pct` contains the substring "value", so an inr-first check typed a
# percentage as money — in the engine whose whole purpose is to stop exactly that.
# Narrower suffixes must be tested before the broad money keywords.
from __future__ import annotations

from app.ai.deep.engine import _kind, _fmt


def test_percent_columns_are_not_money_even_when_named_value():
    assert _kind("value_share_pct", [{"value_share_pct": 85.0}]) == "pct"
    assert _fmt(85.0, "pct") == "85.0%"


def test_lead_time_is_days_not_money():
    assert _kind("avg_lead_time_days", [{"avg_lead_time_days": 5}]) == "days"
    assert _kind("median_lead_time_days", [{"median_lead_time_days": 3}]) == "days"


def test_real_money_columns_still_read_as_money():
    for col in ("revenue", "total_spend", "line_value", "clean_median_price", "purchase_value"):
        assert _kind(col, [{col: 1.0}]) == "inr", col
    assert _fmt(2977736408.82, "inr") == "₹297.77 Cr"


def test_text_and_counts_are_left_alone():
    assert _kind("vendor_name", [{"vendor_name": "Vardhman"}]) == "text"
    assert _kind("lines", [{"lines": 12}]) == "num"
    assert _fmt(45223, "num") == "45,223"


def test_missing_values_render_as_a_dash_not_zero():
    # a blank cell reported as 0 is a wrong number, not a missing one
    assert _fmt(None, "inr") == "—"


# ── presentation defects found in a live deep brief ───────────────────────────
# The KEYTRUDA "sales trend" brief shipped with: a chart whose every bar read "2,026"
# (the YEAR column typed as a measure and charted), years printed as "2,025", months in
# alphabetical order (April, August, December…), and — worst — the stock-change table
# displayed underneath, which is the exact evidence the brief had just RULED OUT.
from app.ai.deep.engine import _order_rows


def test_year_is_an_identifier_not_a_measure():
    # this is what charted twelve identical 2,026 bars
    assert _kind("year", [{"year": 2026}]) == "id"
    assert _fmt(2026, "id") == "2026"          # never "2,026"


def test_other_identifier_columns_are_excluded_too():
    for c in ("month_num", "material", "po_no", "vendor_code"):
        assert _kind(c, [{c: 1}]) == "id", c


def test_a_real_measure_beside_a_year_is_still_a_measure():
    assert _kind("total_stock_change", [{"total_stock_change": 446}]) == "num"


def test_months_come_back_in_time_order_not_alphabetical():
    res = {"columns": ["year", "month", "total_stock_change"],
           "rows": [{"year": 2026, "month": "April", "total_stock_change": 284},
                    {"year": 2025, "month": "December", "total_stock_change": 446},
                    {"year": 2026, "month": "January", "total_stock_change": 456},
                    {"year": 2026, "month": "February", "total_stock_change": 365}]}
    got = [(r["year"], r["month"]) for r in _order_rows(res)["rows"]]
    assert got == [(2025, "December"), (2026, "January"), (2026, "February"), (2026, "April")]


def test_ordering_leaves_non_month_results_untouched():
    res = {"columns": ["vendor_name", "spend"],
           "rows": [{"vendor_name": "B", "spend": 2}, {"vendor_name": "A", "spend": 1}]}
    assert _order_rows(res)["rows"] == res["rows"]


# ── the deep engine's own Plant→Hospital guarantee ───────────────────────────
# Fast mode learned this the hard way (see tests/test_output_sanitizer.py: a live persona
# audit found "Plant" leaking into the model's own phrasing in roughly half of sessions,
# once immediately after an explicit user correction — a prompt instruction would not make
# it stick). Deep mode's synthesis prompt says "say hospital, never plant" and its first
# real brief said "At plant AH01, purchasing peaked". Instructions are preferences;
# this is the guarantee.
from app.ai.deep.engine import _hospitalise


def test_plant_becomes_hospital_preserving_case_and_plurality():
    assert _hospitalise("At plant AH01") == "At hospital AH01"
    assert _hospitalise("Plants HC05 and HC06") == "Hospitals HC05 and HC06"
    assert _hospitalise("PLANT") == "Hospital"


def test_it_does_not_maul_words_that_merely_contain_plant():
    assert _hospitalise("implant surgery") == "implant surgery"
    assert _hospitalise("transplant unit") == "transplant unit"


def test_hospital_codes_and_empty_input_survive():
    assert _hospitalise("HC05 is the busiest") == "HC05 is the busiest"
    assert _hospitalise("") == ""
