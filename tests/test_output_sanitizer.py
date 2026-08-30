# tests/test_output_sanitizer.py — the deterministic Plant->Hospital / debug-marker
# filter every chatbot answer passes through once, in orchestrator.py. See
# _sanitize_prose's docstring: the live persona audit found "Plant" leaking into the
# model's own unprompted phrasing (opening prose, chart titles, table column labels) in
# roughly half of sessions, including once after an EXPLICIT user correction — a prompt
# instruction alone kept failing to make this stick turn over turn. This is the
# regression guard for the deterministic fix, not the prompt.
from __future__ import annotations

from app.ai.orchestrator import _plant_to_hospital, _sanitize_prose, _format_kpi_payload


def test_plant_to_hospital_preserves_case_and_plurality():
    assert _plant_to_hospital("plant") == "hospital"
    assert _plant_to_hospital("Plant") == "Hospital"
    assert _plant_to_hospital("plants") == "hospitals"
    assert _plant_to_hospital("Plants") == "Hospitals"


def test_plant_to_hospital_mid_sentence():
    assert _plant_to_hospital("Inventory data is plant-based across 51 plants.") == \
        "Inventory data is hospital-based across 51 hospitals."
    assert _plant_to_hospital("The most underperforming plants are HT12 and HT15.") == \
        "The most underperforming hospitals are HT12 and HT15."
    assert _plant_to_hospital("All Plants shows healthy stock.") == \
        "All Hospitals shows healthy stock."


def test_plant_to_hospital_does_not_touch_hospital_codes():
    # HC05, HT12 etc. must survive untouched -- only the WORD "plant" is a target.
    assert _plant_to_hospital("HC05 is doing well.") == "HC05 is doing well."


def test_plant_to_hospital_none_and_empty_are_safe():
    assert _plant_to_hospital(None) is None
    assert _plant_to_hospital("") == ""


def test_sanitize_prose_strips_leaked_scope_marker():
    text = "HC05 turnover is 80.56x. [active scope: inventory turnover ratio; plant=HC05]"
    out = _sanitize_prose(text)
    assert "[active scope" not in out
    assert "80.56x" in out


def test_sanitize_prose_applies_plant_filter_after_stripping_marker():
    text = "Break this down by plant. [active scope: fill rate; plant=All Plants]"
    out = _sanitize_prose(text)
    assert "[active scope" not in out
    assert "hospital" in out.lower()
    assert "plant" not in out.lower()


# --- _format_kpi_payload -----------------------------------------------------------
# Live bug caught by asking the deployed AI Analyst a real question: get_kpi's raw
# payload (bands[0]['value'] == 300210253.64, real rupees) reached the model
# UNFORMATTED, so the model did its own Cr conversion in prose and divided by 1e6
# instead of 1e7 -- every band in that answer was off by exactly 10x ("₹300.21 Cr" for
# a real ₹30.02 Cr), while the deterministic chart/table built from the same raw number
# via _fmt() stayed correct. Canonical (get_kpi-only) answers skip the LLM auditor
# entirely, so nothing else was there to catch a wrong PROSE number derived from a
# right raw one. This is the regression guard for pre-formatting the payload before
# it ever reaches the model, so it only ever has to quote, never convert.
def test_format_kpi_payload_formats_nested_money_field():
    raw = {"bands": [{"key": "critical", "label": "< 15 days", "count": 21558, "value": 300210253.64}]}
    out = _format_kpi_payload(raw)
    assert out["bands"][0]["value"] == "₹30.02 Cr"
    # non-money fields must stay untouched (count is a plain number, not a rupee value)
    assert out["bands"][0]["count"] == 21558


def test_format_kpi_payload_formats_days_and_percent_and_leaves_labels():
    raw = {"totals": {"median_doh": 16.574645047467023, "fresh_pct": 71.96, "skus": 18080},
           "label": "not a number"}
    out = _format_kpi_payload(raw)
    assert out["totals"]["median_doh"] == "17 d"
    assert out["totals"]["fresh_pct"] == "72.0%"
    assert out["totals"]["skus"] == 18080  # plain count, no unit to convert
    assert out["label"] == "not a number"


def test_format_kpi_payload_handles_none_and_non_numeric_leaves():
    raw = {"value": None, "name": "M065-INJECTIONS", "nested": {"cost": None}}
    out = _format_kpi_payload(raw)
    assert out["value"] is None
    assert out["name"] == "M065-INJECTIONS"
    assert out["nested"]["cost"] is None


# ── deterministic number verification ────────────────────────────────────────
# The LLM auditor is measurably unreliable on figure errors: prompted 5x with a
# mislabelled metric it caught it only 2/5, and it costs ~2.1s of serial latency. Every
# figure the model sees has already been rendered by _fmt(), so "is this number supported"
# is decidable by string membership. These lock that in — especially the real 10x bug that
# reached production (₹300.21 Cr printed for a ₹30.02 Cr figure) in a get_kpi-only turn
# that the audit was skipping entirely.
from app.ai.orchestrator import _unsupported_numbers, _canon_num, _evidence_numbers

_RES = [{"columns": ["label", "value", "share"],
         "rows": [{"label": "critical", "value": 300210253.64, "share": 49.6},
                  {"label": "low", "value": 55290098.93, "share": 9.1}],
         "row_count": 2}]


def test_catches_the_real_10x_error():
    bad = _unsupported_numbers("Critical stock is ₹300.21 Cr (49.6%) and low is ₹55.29 Cr.", _RES)
    assert "₹300.21 Cr" in bad and "₹55.29 Cr" in bad


def test_passes_the_correct_figures():
    assert _unsupported_numbers("Critical is ₹30.02 Cr (49.6%) and low is ₹5.53 Cr (9.1%).", _RES) == []


def test_money_is_zero_tolerance_even_when_most_figures_are_right():
    # an earlier draft used a flat tolerance of 2 and let the 10x bug through, because that
    # answer contained exactly two bad figures. A rupee magnitude is never "derived".
    bad = _unsupported_numbers("Critical ₹30.02 Cr, low ₹5.53 Cr, total ₹999.00 Cr.", _RES)
    assert bad == ["₹999.00 Cr"]


def test_derived_percentages_do_not_false_flag():
    # an analyst legitimately says "up 25%" without 25 appearing in any cell; a guard that
    # bounces those gets switched off, so percentages carry a small tolerance
    assert _unsupported_numbers("Critical is ₹30.02 Cr, up 25% on the ₹5.53 Cr low band.", _RES) == []


def test_prose_without_numbers_is_never_flagged():
    assert _unsupported_numbers("Most of the stock sits in the critical band.", _RES) == []
    assert _unsupported_numbers("", _RES) == []
    assert _unsupported_numbers("₹30.02 Cr", []) == []      # no evidence => nothing to check against


def test_canonicalisation_ignores_formatting_noise():
    assert _canon_num("₹ 30.02 Cr") == _canon_num("₹30.02 Cr") == "30.02cr"
    assert _canon_num("72.0%") == _canon_num("72%") == "72%"   # model rounds legitimately
    assert _canon_num("1,234") == "1234"


def test_evidence_set_is_built_from_formatted_values():
    ev = _evidence_numbers(_RES)
    assert "30.02cr" in ev and "5.53cr" in ev and "49.6%" in ev
