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
