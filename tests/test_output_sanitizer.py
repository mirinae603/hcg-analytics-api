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


def test_derived_figures_are_not_flagged():
    # THE regression that mattered. The first version asked "does every figure appear in
    # the evidence?" and so flagged every legitimately-derived number — a margin computed
    # as revenue minus cost, a total, an average. Each flag bounced the answer for a rewrite
    # and the retry ran up to three times: that is the repeated "Correcting the analysis"
    # loop seen in production, ~6s added to a turn for nothing. A guard that fires on
    # correct answers is worse than no guard.
    assert _unsupported_numbers("Critical ₹30.02 Cr and low ₹5.53 Cr, ₹35.55 Cr between them.", _RES) == []
    assert _unsupported_numbers("Together they account for ₹35.55 Cr, averaging ₹17.78 Cr.", _RES) == []


def test_only_power_of_ten_scale_errors_are_flagged():
    # a figure unrelated to the evidence is NOT this check's job — that is the LLM
    # auditor's. This one answers a narrow, decidable question: right digits, wrong scale.
    assert _unsupported_numbers("Total is ₹999.00 Cr.", _RES) == []
    assert _unsupported_numbers("Critical is ₹300.21 Cr.", _RES) == ["₹300.21 Cr"]     # 10x
    assert _unsupported_numbers("Critical is ₹3002.10 Cr.", _RES) == ["₹3002.10 Cr"]   # 100x


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


# ── the answer must be the prose the model actually wrote ──────────────────────
# PRESENT_TOOL's `answer` argument is DEPRECATED: the model is told to write the answer as
# message text and call present() only for the visuals. But the shipped answer was read
# straight back off that deprecated argument (`present_args.get("answer")`), so whenever the
# model followed its instruction the turn went out with a chart, a table and no words.
# Reproduced end to end against the local backend: session 13's row persisted as
# {"text": ""} while session 12's identical question, answered in the older style, came
# through at 4,864 characters. These lock the resolution order `cand_ans` implements.
def _resolve(streamed_prose: str, last_prose: str, present_answer: str) -> str:
    """Mirror of the orchestrator's `cand_ans` precedence, kept in one place."""
    return (streamed_prose or last_prose or present_answer or "").strip()


def test_streamed_prose_wins_over_the_deprecated_argument():
    assert _resolve("Spend is concentrated in one vendor.", "", "stale") == \
        "Spend is concentrated in one vendor."


def test_prose_from_an_earlier_round_is_not_lost():
    # the model writes the prose in one round and calls present() in the next, so the
    # per-round `streamed_prose` is empty by the time the call is accepted
    assert _resolve("", "Written a round earlier.", "") == "Written a round earlier."


def test_falls_back_to_the_argument_when_the_model_used_it():
    assert _resolve("", "", "Put it in the tool call after all.") == "Put it in the tool call after all."


def test_empty_everywhere_stays_empty_rather_than_inventing_text():
    assert _resolve("", "", "") == ""
    assert _resolve("   ", "", "") == ""
