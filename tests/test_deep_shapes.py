# tests/test_deep_shapes.py — what a good answer LOOKS LIKE, per kind of question.
#
# The engine could explore the warehouse competently and still write this:
#   "The purchasing trend shows variability across hospitals and months, with examples
#    including 80 units purchased by HC05 in December 2025... WHAT DRIVES IT: this data is
#    grouped by year, month, and hospital. The material ID for Keytruda is 101313."
# Two cells read aloud plus a description of the SQL. Nothing in the engine knew what a
# TREND answer owes the reader, so it described the table it happened to get.
from __future__ import annotations

from app.ai.deep import shapes
from app.ai.deep.engine import _measure_disclosure


# the real result shape that produced that brief: hospital BY month, not a series
GRID = {"columns": ["year", "month", "hospital", "total_qty"],
        "rows": [{"year": 2025, "month": "December", "hospital": "HC05", "total_qty": 80},
                 {"year": 2025, "month": "December", "hospital": "HM01", "total_qty": 75},
                 {"year": 2026, "month": "January", "hospital": "HC05", "total_qty": 94},
                 {"year": 2026, "month": "January", "hospital": "HM01", "total_qty": 63},
                 {"year": 2026, "month": "April", "hospital": "HC05", "total_qty": 50}]}


def test_trend_collapses_a_grid_into_one_series():
    # a trend is ONE line. Any other dimension has to be summed away before it is one —
    # otherwise the only honest thing to say about it is "it varies", which is what the
    # engine used to say.
    d = shapes.derive_trend(GRID)
    assert [p["period"] for p in d["series"]] == ["2025 December", "2026 January", "2026 April"]
    assert d["series"][0]["value"] == 155.0    # 80 + 75, not two separate rows


def test_trend_states_direction_and_magnitude_not_variability():
    d = shapes.derive_trend(GRID)
    assert d["direction"] == "falling"
    assert round(d["change_pct_first_to_last"], 1) == -67.7
    assert d["peak"]["period"] == "2026 January"
    assert d["trough"]["period"] == "2026 April"


def test_trend_needs_at_least_two_periods_to_be_a_trend():
    one = {"columns": ["month", "qty"], "rows": [{"month": "January", "qty": 5}]}
    assert shapes.derive_trend(one) == {}


def test_identifier_columns_are_never_treated_as_the_measure():
    # `year` is numeric and was previously picked as the value, producing a chart of
    # twelve identical 2,026 bars
    d = shapes.derive_trend(GRID)
    assert d["measure"] == "total_qty"


def test_ranking_reports_concentration_not_just_the_list():
    res = {"columns": ["vendor_name", "spend"],
           "rows": [{"vendor_name": "Vardhman", "spend": 297.77},
                    {"vendor_name": "Wipro", "spend": 21.10},
                    {"vendor_name": "Advanced Medtech", "spend": 15.00},
                    {"vendor_name": "Akshay", "spend": 13.41}]}
    d = shapes.derive_ranking(res)
    assert d["top"][0]["label"] == "Vardhman"
    # renamed: it is the share of the rows RETURNED, not of the company. Calling it
    # `share_pct` is how "100% of total procurement" got written from a one-row result.
    assert 85 < d["top1_share_of_returned_pct"] < 87
    assert d["tail_n"] == 1
    assert "rows returned" in d["share_note"]


def test_every_shape_declares_what_the_answer_owes():
    for name, sh in shapes.SHAPES.items():
        assert sh["answer_must"], name
        assert sh["slots"], name
        assert any(s.get("required") for s in sh["slots"]), name


# ── the substitution disclosure, made deterministic ──────────────────────────
# There is no material-by-month SALES grain, so a "sales trend" is answered from
# PURCHASING. That substitution is fine; labelling it "Sales Trend" is a wrong answer
# however good the arithmetic. The prompt has asked for this disclosure since the first
# version and the model supplies it about half the time — the same reliability that
# Plant->Hospital taught us twice already.
def test_discloses_when_the_measure_was_substituted():
    findings = [{"sql": "SELECT year, month, SUM(total_value_wo_tax) FROM fact_po WHERE material='101313' GROUP BY 1,2"}]
    msg = _measure_disclosure("get me the sales trend of KEYTRUDA", findings)
    assert "no sales figure" in msg and "PURCHASING" in msg


def test_silent_when_the_measure_matches():
    findings = [{"sql": "SELECT revenue FROM sales_by_material WHERE material='101313'"}]
    assert _measure_disclosure("sales for keytruda", findings) == ""


def test_silent_when_the_question_names_no_measure():
    findings = [{"sql": "SELECT vendor_name FROM fact_po WHERE material='101313'"}]
    assert _measure_disclosure("who supplies keytruda", findings) == ""


def test_consumption_substitution_is_caught_too():
    # generalises beyond the one case that prompted it
    findings = [{"sql": "SELECT month, SUM(qty) FROM fact_consumption WHERE material='101313' GROUP BY 1"}]
    assert "CONSUMPTION" in _measure_disclosure("monthly sales of keytruda", findings)


def test_disclosure_judges_the_headline_not_every_table_touched():
    # A turn that reads the sales TOTAL for context while building its monthly series from
    # PURCHASING was staying silent, because "sales" appeared somewhere in the turn — while
    # the number in the first sentence, the one the reader takes away, was purchasing.
    findings = [
        {"sql": "SELECT revenue FROM sales_by_material WHERE material='101313'", "purpose": "total"},
        {"sql": "SELECT year,month,SUM(monthly_purchase_value) FROM kpi_monthly_purchase_value "
                "WHERE material='101313' GROUP BY 1,2", "purpose": "monthly series"},
    ]
    headline = findings[1]
    assert "PURCHASING" in _measure_disclosure("sales trend of KEYTRUDA", findings, headline)
    # and stays silent when the headline really is the measure asked for
    assert _measure_disclosure("sales trend of KEYTRUDA", findings, findings[0]) == ""


def test_derived_facts_are_formatted_in_their_own_units():
    # shapes.py works in raw floats so it stays testable, but handing those to the writer
    # produced "sales started at 80,663,366" — a rupee figure as a bare integer, in the
    # very block whose job is to stop the model doing its own arithmetic. And a percent key
    # that does not END in "_pct" (change_pct_first_to_last) fell through to money: "₹-4".
    from app.ai.deep.engine import _format_derived
    out = _format_derived([{
        "measure": "monthly_purchase_value",
        "first": {"period": "2025 December", "value": 80663365.9},
        "change_pct_first_to_last": -3.54,
        "swing_pct_trough_to_peak": 46.5,
    }])[0]
    assert out["first"]["value"] == "₹8.07 Cr"
    assert out["change_pct_first_to_last"] == "-3.5%"
    assert out["swing_pct_trough_to_peak"] == "46.5%"


def test_a_quantity_measure_is_not_rendered_as_money():
    from app.ai.deep.engine import _format_derived
    out = _format_derived([{"measure": "total_qty", "peak": {"period": "Jan", "value": 45223.0}}])[0]
    assert out["peak"]["value"] == "45,223"


def test_a_single_row_result_carries_a_warning_instead_of_a_share():
    # Returning nothing for a one-row result left the model to compute the share itself,
    # which produced "100% of the total procurement value of ₹649.91 Cr is sourced from a
    # single vendor" (the true leader is 45.8%) and "GLASS PAPER accounts for 100.0% of
    # out-of-stock demand". Both came from dividing a number by itself.
    d = shapes.derive_ranking({"columns": ["vendor_name", "spend"],
                               "rows": [{"vendor_name": "Vardhman", "spend": 6499100000}]})
    assert d["n"] == 1
    assert "share_pct" not in d and "top1_share_of_returned_pct" not in d
    assert "no denominator" in d["share_note"]
    assert d["top"][0]["value"] == 6499100000.0        # the value itself is still reported


def test_two_rows_still_refuse_to_produce_a_share():
    d = shapes.derive_ranking({"columns": ["k", "v"], "rows": [{"k": "a", "v": 2}, {"k": "b", "v": 1}]})
    assert "top1_share_of_returned_pct" not in d
    assert "no denominator" in d["share_note"]


# ── shares only mean something for additive measures ─────────────────────────
def test_no_share_is_computed_for_a_rate_or_average():
    # A live brief said "these top three vendors account for 43.2% of the total lead time".
    # Lead time is days: the sum of everyone's days is not a quantity anyone holds, and a
    # share of it is arithmetic without meaning. It reads as precision and carries none.
    d = shapes.derive_ranking({"columns": ["vendor_name", "avg_lead_time"],
                               "rows": [{"vendor_name": "SP", "avg_lead_time": 28},
                                        {"vendor_name": "Axon", "avg_lead_time": 23},
                                        {"vendor_name": "Medicare", "avg_lead_time": 23},
                                        {"vendor_name": "Quick", "avg_lead_time": 2}]})
    assert "top1_share_of_returned_pct" not in d
    assert "not an additive quantity" in d["share_note"]
    # what IS useful about a rate: the spread
    assert d["spread"]["highest"]["label"] == "SP"
    assert d["spread"]["lowest"]["value"] == 2.0
    assert d["spread"]["median"] == 23.0


def test_additive_measures_still_get_their_concentration():
    d = shapes.derive_ranking({"columns": ["vendor_name", "spend"],
                               "rows": [{"vendor_name": "A", "spend": 100},
                                        {"vendor_name": "B", "spend": 50},
                                        {"vendor_name": "C", "spend": 25}]})
    assert round(d["top1_share_of_returned_pct"], 1) == 57.1


def test_percentage_and_score_columns_are_treated_as_rates_too():
    for col in ("margin_pct", "fill_rate", "health_score", "median_lead_time_days"):
        d = shapes.derive_ranking({"columns": ["k", col],
                                   "rows": [{"k": "a", col: 9}, {"k": "b", col: 5}, {"k": "c", col: 1}]})
        assert "top1_share_of_returned_pct" not in d, col
