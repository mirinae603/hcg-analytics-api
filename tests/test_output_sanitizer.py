# tests/test_output_sanitizer.py — the deterministic Plant->Hospital / debug-marker
# filter every chatbot answer passes through once, in orchestrator.py. See
# _sanitize_prose's docstring: the live persona audit found "Plant" leaking into the
# model's own unprompted phrasing (opening prose, chart titles, table column labels) in
# roughly half of sessions, including once after an EXPLICIT user correction — a prompt
# instruction alone kept failing to make this stick turn over turn. This is the
# regression guard for the deterministic fix, not the prompt.
from __future__ import annotations

from app.ai.orchestrator import _plant_to_hospital, _sanitize_prose


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
