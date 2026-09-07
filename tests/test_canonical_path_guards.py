# tests/test_canonical_path_guards.py — the path that keeps escaping the guards.
#
# Canonical KPI findings are built directly in answer(), not through investigate(), so every
# check wired into investigate's return skips them. This has now bitten twice:
#
#   1. scope — "Bangalore hospitals spend ₹649.91 Cr", the whole 51-site network.
#   2. placeholders — "the largest spend category is Uncategorized", one run AFTER the
#      withdrawal guard was verified passing, because the row arrived by this path.
#
# Both are fixed at the source. These tests exist so a third does not happen quietly.
from __future__ import annotations

import inspect

from app.ai.deep import engine


def _answer_src() -> str:
    return inspect.getsource(engine.answer)


def test_every_canonical_kpi_result_is_placeholder_checked():
    src = _answer_src()
    kpi_sites = src.count("_kpi_rows(out)")
    guarded = src.count("sink_placeholders")
    assert kpi_sites >= 2, "expected both canonical construction sites"
    assert guarded >= kpi_sites, (
        f"{kpi_sites} canonical results built but only {guarded} placeholder checks — a "
        f"canonical finding is escaping the guard again")


def test_the_canonical_call_passes_the_question_for_scope_checking():
    # without the question, _kpi_scope_mismatch cannot tell a city question from a
    # network one, and the unscoped total goes out labelled as four hospitals
    assert "tools.get_kpi(kpi_key, question=query)" in _answer_src()


def test_sink_placeholders_actually_withdraws_from_a_kpi_shaped_result():
    from app.ai.deep.sanity import sink_placeholders
    res = {"columns": ["name", "value"],
           "rows": [{"name": "Uncategorized", "value": 1_733_085_192.5},
                    {"name": "ANTINEOPLASTIC", "value": 844_500_000.0}],
           "row_count": 2}
    assert sink_placeholders(res) is True
    assert res["rows"][0]["name"] == "ANTINEOPLASTIC"
    assert res["row_count"] == 1
    assert "Uncategorized" in res["excluded_note"]
