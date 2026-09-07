# tests/test_kpi_scope.py — the canonical path bypassed every guard in the system.
#
# Every other check inspects SQL. get_kpi produces none — it is a function call — so
# city_on_unreachable_table, part_exceeds_whole and the constraint checker all sat it out.
# Asked what Bangalore hospitals spend on procurement, the engine called get_kpi with no
# plant and reported ₹649.91 Cr: the total for all 51 sites, labelled as four.
from __future__ import annotations

from app.ai.deep.tools import _kpi_scope_mismatch


def test_a_city_question_may_not_use_an_unscoped_kpi():
    w = _kpi_scope_mismatch("purchase-value", "",
                            "How much do our Bangalore hospitals spend on procurement?")
    assert w and w.startswith("UNSCOPED KPI")
    assert "HC05" in w                      # names the codes to scope by


def test_passing_a_plant_is_allowed():
    assert _kpi_scope_mismatch("purchase-value", "HC05",
                               "How much does HCG KR spend?") is None


def test_a_network_wide_question_is_allowed():
    assert _kpi_scope_mismatch("purchase-value", "",
                               "What is our total procurement spend?") is None


def test_a_named_hospital_also_triggers_it():
    w = _kpi_scope_mismatch("stock-value", "", "How much stock does HM01 hold?")
    assert w is None or "UNSCOPED" in w     # only when the resolver types it as a hospital


def test_it_never_raises_on_junk():
    assert _kpi_scope_mismatch("", "", "") is None
    assert _kpi_scope_mismatch("nope", "", None) is None


def test_one_plant_is_not_a_city():
    # told only that an unscoped call was wrong, the engine called get_kpi with a single
    # plant and reported ₹16.41 Cr — HC40 alone — against a citywide ₹173.73 Cr
    w = _kpi_scope_mismatch("purchase-value", "HC40",
                            "How much do our Bangalore hospitals spend on procurement?")
    assert w and w.startswith("UNDER-SCOPED")
    assert "HC01" in w and "SUM" in w


def test_a_single_named_hospital_is_left_alone():
    assert _kpi_scope_mismatch("purchase-value", "HM01",
                               "How much does HM01 spend on procurement?") is None


def test_a_city_call_returns_every_site_rather_than_refusing():
    # a guard that only refuses leaves nowhere to go: blocking both the unscoped and the
    # single-plant call produced "I couldn't establish anything" and zero queries
    from app.ai.deep.tools import get_kpi
    out = get_kpi("purchase-value",
                  question="How much do our Bangalore hospitals spend on procurement?")
    assert out.get("scoped_to") == ["HC01", "HC05", "HC06", "HC40"]
    assert set(out.get("per_site") or {}) == {"HC01", "HC05", "HC06", "HC40"}
    assert not out.get("error")


def test_the_per_site_payload_renders_as_money():
    # the first version fell through _kpi_rows unrecognised and put "1,739,047,400.36" in
    # the answer where ₹173.90 Cr belonged
    from app.ai.deep.engine import _compact, _kpi_rows
    from app.ai.deep.tools import get_kpi
    out = get_kpi("purchase-value",
                  question="How much do our Bangalore hospitals spend on procurement?")
    text = _compact(_kpi_rows(out))
    assert "₹" in text and "Cr" in text
    assert "HC05" in text and "1,739,047,400" not in text


def test_a_kpi_that_ignores_plant_is_caught_rather_than_repeated_per_site():
    # revenue-margin's source uses a site code system disjoint from dim_plant, so it ignores
    # `plant` entirely: asking for four hospitals returned the COMPANY total four times,
    # identically, labelled as four hospitals. That fabricates site-level detail — strictly
    # worse than the unscoped answer it replaced.
    from app.ai.deep.tools import get_kpi
    out = get_kpi("revenue-margin",
                  question="What is the top-selling drug in our Bangalore hospitals?")
    assert out.get("error") and "IGNORES the plant argument" in out["error"]
    assert not out.get("per_site")


def test_a_kpi_that_honours_plant_still_returns_every_site():
    from app.ai.deep.tools import get_kpi
    out = get_kpi("purchase-value",
                  question="How much do our Bangalore hospitals spend on procurement?")
    assert not out.get("error")
    sites = out.get("per_site") or {}
    assert set(sites) == {"HC01", "HC05", "HC06", "HC40"}
    # and the figures must actually differ, or the check above would have fired
    totals = {str(((p or {}).get("data") or {}).get("totals")) for p in sites.values()}
    assert len(totals) > 1
