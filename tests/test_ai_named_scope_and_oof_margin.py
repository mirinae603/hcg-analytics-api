# tests/test_ai_named_scope_and_oof_margin.py — two answers a live run got wrong with full
# confidence (12-11-2026), which a spoken answer would only make sound more certain.
#
# 1. "What was formulary compliance at HCG KR Bangalore in August 2026?" was answered for all
#    four Bangalore sites (HC01, HC05, HC06, HC40): "Bangalore" inside KR's name was read as
#    the city, so the brief listed four sites, the scope guard refused plant='HC05' as
#    under-scoped, and the headline moved between runs (pooled 98.51% vs KR's 98.42%).
# 2. "Top 10 out-of-formulary brands by margin missed in Aug 2026" ranked formulary-
#    compliance's out-of-formulary drug COST, worded it as margin missed, and showed 16 rows.
#
# Offline: the hospital tests read the committed hospital map and the local warehouse (as
# tests/test_resolve.py does); the report tests use test_spend_oof_purchase_api's invented
# purchases, and the golden tests the committed purchase tables, skipped when not built.
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.ai import hcg_reports as H
from app.ai import hospitals, kpi_registry as kr, resolve, routing
from app.ai import orchestrator as orch
from app.ai.deep import tools as deep_tools
from app.api import spend
from app.core.config import settings
from tests.test_orchestrator_fixes import FakeClient, offline  # noqa: F401  (offline is a fixture)
from tests.test_spend_oof_purchase_api import AUG, DR, JUL, KR, data  # noqa: F401  (data is a fixture)

COMPLIANCE_AT_KR = "What was formulary compliance at HCG KR Bangalore in August 2026?"


# ── 1. a named hospital is that hospital ──────────────────────────────────────
@pytest.mark.parametrize("named", ["HCG KR Bangalore", "KR Bangalore", "KR, Bangalore", "HCG KR, Bangalore",
                                   "kr bangalore", "KR in Bangalore"])
def test_kr_bangalore_is_one_hospital_and_not_the_city(named):
    q = f"What was formulary compliance at {named} in August 2026?"
    assert [codes for _a, codes in hospitals.mentioned(q) if len(codes) == 1] == [("HC05",)]
    assert resolve._city_only_addresses_a_hospital(q, "Bangalore")
    r = resolve.resolve(q)
    assert "Bangalore" not in r["cities"], r["cities"]
    assert not any(f["token"] == "bangalore" for f in r["families"]), r["families"]
    # KR's own figure is the right call; asking without a plant is told to use KR alone
    assert deep_tools._kpi_scope_mismatch("formulary-compliance", "HC05", q, ratio=True) is None
    told = deep_tools._kpi_scope_mismatch("formulary-compliance", "", q, ratio=True)
    assert ("plant='HC05'" in told or "(HCG KR, Bangalore)" in told) and "MORE THAN ONE" not in told
    assert not any(code in told for code in ("HC01", "HC06", "HC40"))


def test_the_brief_for_kr_names_kr_alone():
    b = resolve.brief(COMPLIANCE_AT_KR)
    assert "SAP plant HC05" in b
    assert "is a CITY" not in b and "HC01" not in b and "HC06" not in b and "HC40" not in b
    assert "Cover the whole family" not in b


def test_a_city_alone_still_covers_every_site_in_it():
    q = "What was formulary compliance in Bangalore in August 2026?"
    assert not resolve._city_only_addresses_a_hospital(q, "Bangalore")
    r = resolve.resolve(q)
    assert "Bangalore" in r["cities"]
    codes = {s.split(" ")[0] for s in r["city_hospitals"]["Bangalore"]}
    assert {"HC05", "HC06", "HC40"} <= codes
    # one Bangalore hospital does not answer for the city
    assert deep_tools._kpi_scope_mismatch("formulary-compliance", "HC05", q, ratio=True).startswith("UNDER-SCOPED")


@pytest.mark.parametrize("named, code", [("HCG DR, Bangalore", "HC06"), ("North Bangalore", "HC40"),
                                         ("HCG DR Bangalore", "HC06")])
def test_another_named_bangalore_site_is_that_site(named, code):
    q = f"formulary compliance at {named} in August 2026"
    r = resolve.resolve(q)
    assert "Bangalore" not in r["cities"] and "North" not in r["cities"]
    assert deep_tools._kpi_scope_mismatch("formulary-compliance", code, q, ratio=True) is None


def test_a_city_named_beside_a_hospital_stays_a_city():
    # the user asks about KR AND the rest of the city: both scopes are real
    q = "compare KR with the rest of Bangalore"
    assert not resolve._city_only_addresses_a_hospital(q, "Bangalore")
    assert "Bangalore" in resolve.resolve(q)["cities"]
    # "in" joins a hospital to its OWN city only
    assert not resolve._city_only_addresses_a_hospital("KR in Chennai", "Chennai")


def test_spans_give_the_longest_name_its_words():
    got = hospitals.spans("compliance at HCG KR, Bangalore and DR")
    assert [(alias, codes) for _s, _e, alias, codes in got] == [("KR BANGALORE", ("HC05",)), ("DR", ("HC06",))]
    s, e, _a, _c = got[0]
    assert "compliance at HCG KR, Bangalore and DR"[s:e] == "KR, Bangalore"
    assert [c for _s, _e, _a, c in hospitals.spans("spend in Bangalore")] == [("HC05", "HC06", "HC40")]
    assert hospitals.mentioned("KR vs KR Bangalore vs KR") == [("KR", ("HC05",)), ("KR BANGALORE", ("HC05",))]


# ── 2. out-of-formulary margin missed: the right measure ──────────────────────
TOP10 = "Top 10 out-of-formulary brands by margin missed in Aug 2026 as a table"


def test_the_question_is_locked_to_the_oof_margin_report():
    locks = routing._locks_for(TOP10)
    assert len(locks) == 1 and 'get_kpi("oof-margin-loss")' in locks[0]
    assert routing._locks_for("OOF margin loss at KR") and routing._locks_for("margin lost on OOF brands")
    # out-of-formulary SPEND is formulary-compliance's figure: no lock
    assert not routing._locks_for("top 10 out-of-formulary brands by spend in Aug 2026")
    assert not routing._locks_for("cash and TPA margin loss at KR")


def test_the_brief_says_which_report_holds_oof_margin():
    notes = [n for pat, n in resolve._KEYWORD_NOTES if __import__("re").search(pat, TOP10, __import__("re").I)]
    assert any("get_kpi('oof-margin-loss')" in n and "NOT formulary-compliance" in n for n in notes)
    assert "oof-margin-loss" in resolve._REPORTS_NOTE


def test_the_registry_offers_it_and_tells_it_apart_from_spend():
    e = kr.KPI_REGISTRY["oof-margin-loss"]
    assert e["fn"] is H.oof_margin_loss and e["needs"] == "purchase" and e["ratio"] is True
    assert set(e["args"]) == {"month", "segment", "view", "material", "top"}
    schema = kr.tool_schema()["function"]
    assert "oof-margin-loss" in schema["parameters"]["properties"]["kpi"]["enum"]
    assert "oof-margin-loss" in kr.KPI_REGISTRY["formulary-compliance"]["about"]
    assert "not margin missed" in kr.KPI_REGISTRY["formulary-compliance"]["hint"]
    assert "oof-margin-loss" in orch.SYSTEM


def test_every_report_view_can_be_sent():
    # the view argument is an enum: a view left out of it (vendors was) cannot be asked for
    sendable = set(kr.tool_schema()["function"]["parameters"]["properties"]["view"]["enum"])
    for views in (H.FORMULARY_VIEWS, H.BRAND_RULE_VIEWS, H.ATTIS_VIEWS, H.CASH_VIEWS, H.OOF_VIEWS):
        assert set(views) <= sendable, set(views) - sendable
    assert set(H.RANKED_VIEWS) <= sendable


def test_summary_is_the_dashboards_net_margin_missed(data):  # noqa: F811
    t = H.oof_margin_loss(Plant="KR", month="Aug")["totals"]
    # test_spend_oof_purchase_api.test_headline_is_hcgs_net_sum: earned 7000, possible 9820, missed 2820
    assert (t["actual_margin_value"], t["potential_margin_value"], t["margin_loss_value"]) == (7000.0, 9820.0, 2820.0)
    assert t["margin_loss_pct"] == 28.72 and t["purchase_value"] == 9100.0
    assert (t["brands"], t["vendors"], t["hospitals"]) == (7, 2, 1)


def test_items_rank_brands_by_margin_missed_and_top_cuts_them(data):  # noqa: F811
    out = H.oof_margin_loss(Plant="HC05", month=AUG, view="items")
    # counted brands most missed first (3000, 600, 320, -1100), then the ones with no suggestion
    assert [r["margin_loss_value"] for r in out["rows"][:4]] == [3000.0, 600.0, 320.0, -1100.0]
    assert out["rows"][0]["item"] == "ONCO OOF A 100MG INJ" and out["rows_total"] == 7
    got = kr.call_kpi("oof-margin-loss", plant="HC05", month=AUG, view="items", top=2)
    assert [r["item"] for r in got["data"]["rows"]] == ["ONCO OOF A 100MG INJ", "PARA OOF B TAB"]
    assert got["data"]["rows_total"] == 7 and "_ignored_args" not in got


def test_other_views_and_pooling(data):  # noqa: F811
    trend = H.oof_margin_loss(Plant="HC05", view="trend")["rows"]
    assert [(r["month"], r["margin_loss_value"]) for r in trend] == [("Jul-26", 1800.0), ("Aug-26", 2820.0)]
    vendors = H.oof_margin_loss(Plant="HC05", month=AUG, view="vendors")["rows"]
    assert [(r["vendor"], r["margin_loss_value"]) for r in vendors] == [("Alpha Pharma", 3000.0),
                                                                        ("Beta Distributors", -180.0)]
    ranking = H.oof_margin_loss(month=AUG, view="hospitals")["rows"]
    assert ranking[0]["plant"] == KR and ranking[-1]["plant"] == DR and ranking[-1]["margin_loss_value"] == -1600.0
    pooled = H.oof_margin_loss(Plant=f"{KR},{DR}", month=AUG)["totals"]
    # amounts added first (2820 - 1600 over 9820 + 2400), the % taken once
    assert (pooled["margin_loss_value"], pooled["potential_margin_value"], pooled["margin_loss_pct"]) == \
        (1220.0, 12220.0, 9.98)
    with pytest.raises(ValueError):
        H.oof_margin_loss(Plant=f"{KR},{DR}", view="items")


def test_purchases_take_no_payer_patient_type_or_doctor(data):  # noqa: F811
    got = kr.call_kpi("oof-margin-loss", plant="HC05", month=AUG, payer="CASH", patient_type="IP", doctor="x")
    assert got["_ignored_args"] == ["doctor", "patient_type", "payer"]
    assert got["data"]["totals"]["margin_loss_value"] == 2820.0


# ── golden: the committed purchase tables against the dashboard and HCG's email ─
def _purchases_built() -> bool:
    return (Path(settings.PURCHASE_DATA_DIR) / "oof_purchase.parquet").exists()


@pytest.mark.skipif(not _purchases_built(), reason="the purchase build's tables are not published here")
def test_golden_oof_margin_is_the_dashboards_figure_and_hcgs_email():
    spend.clear_caches()
    email = Path(settings.REFERENCE_DIR) / "oof_email_figures.csv"
    with open(email, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        ours = kr.call_kpi("oof-margin-loss", plant=r["plant"], month=r["period"])["data"]["totals"]
        page = spend.spend_oof_overview(Plant=r["plant"], Month=r["period"], Type=None, Segment=None,
                                        Category=None, Payer=None, Material=None)["headline"]
        assert ours["margin_loss_value"] == page["margin_loss"], r
        assert abs(ours["margin_loss_value"] - float(r["margin_loss"])) <= 1.0, r   # HCG rounds to the rupee
        assert abs(ours["potential_margin_value"] - float(r["suggested_margin"])) <= 1.0, r
    # every hospital, Aug-26: HCG's Summary total (hcg-oof-purchase-spend-method)
    aug = kr.call_kpi("oof-margin-loss", month="2026-08")["data"]["totals"]
    assert aug["margin_loss_value"] == 3243199.13


@pytest.mark.skipif(not _purchases_built(), reason="the purchase build's tables are not published here")
def test_golden_top_10_brands_by_margin_missed_are_the_pages_top_10():
    spend.clear_caches()
    got = kr.call_kpi("oof-margin-loss", month="Aug 2026", view="items", top=10)["data"]
    page = spend.spend_oof_overview(Plant=None, Month="2026-08", Type=None, Segment=None, Category=None,
                                    Payer=None, Material=None)["top_items"]
    assert len(got["rows"]) == 10
    assert [r["material"] for r in got["rows"]] == [r["material"] for r in page[:10]]
    assert [r["margin_loss_value"] for r in got["rows"]] == [r["margin_loss"] for r in page[:10]]


# ── "top N" means N rows ──────────────────────────────────────────────────────
@pytest.mark.parametrize("q, n", [(TOP10, 10), ("top ten OOF brands", 10), ("the 5 worst hospitals", 5),
                                  ("bottom-3 vendors by margin", 3), ("20 biggest brands", 20),
                                  ("which doctors use the most OOF brands", None),
                                  ("compliance over the last 3 months", None), ("top selling drugs in 2026", None)])
def test_top_n_reads_the_count_a_question_asks_for(q, n):
    assert resolve.top_n(q) == n


# A month is no count: HCG writes "Aug-26", the app "2026-08". Read as counts, these cut every
# ranked table to 26 or 8 rows. Nor is "which one most …" a top 1.
@pytest.mark.parametrize("q, n", [("Aug-26 top vendors by OOF margin missed", None),
                                  ("Jun-26 highest out-of-formulary items", None),
                                  ("for 2026-08 top items", None), ("Aug 26 top vendors", None),
                                  ("26/08 top items", None), ("FY'26 top vendors", None),
                                  ("Which one most often misses margin?", None),
                                  ("top 10 brands in Aug-26", 10), ("in May, 5 biggest brands", 5),
                                  ("top one brand", 1), ("the two most expensive brands", 2),
                                  ("top-10 brands", 10)])
def test_top_n_never_reads_a_date_or_which_one_as_the_count(q, n):
    assert resolve.top_n(q) == n


_RANKED = {"totals": {"margin_loss_value": 1000.0}, "scope": {"plant": "ALL"}, "view": "items",
           "rows": [{"item": f"BRAND {i}", "margin_loss_value": 100.0 - i} for i in range(16)], "rows_total": 461,
           "notes": ["How margin missed is worked out.", "Showing the first 16 of 461 rows."],
           "data_window": "2026-06..2026-08"}


def test_top_rows_cuts_only_ranked_views():
    out, applied = H.top_rows(_RANKED, 10)
    assert applied and [r["item"] for r in out["rows"]] == [f"BRAND {i}" for i in range(10)]
    assert out["rows_total"] == 461 and out["notes"] == ["How margin missed is worked out.",
                                                          "The top 10 of 461 rows, ranked as listed."]
    assert len(_RANKED["rows"]) == 16                            # the handler's payload is not changed
    trend = {**_RANKED, "view": "trend"}
    assert H.top_rows(trend, 3) == (trend, False)
    assert H.top_rows({**_RANKED, "view": "hospitals"}, 3)[1] is False


def test_call_kpi_reports_top_it_could_not_apply_and_refuses_a_bad_one(monkeypatch):
    monkeypatch.setitem(kr.KPI_REGISTRY["oof-margin-loss"], "fn", lambda **kw: {**_RANKED, "view": kw.get("view")})
    assert len(kr.call_kpi("oof-margin-loss", view="items", top="10")["data"]["rows"]) == 10
    assert kr.call_kpi("oof-margin-loss", view="trend", top=10)["_ignored_args"] == ["top"]
    for bad in (0, 51, "ten"):
        with pytest.raises(ValueError):
            kr.call_kpi("oof-margin-loss", view="items", top=bad)


def test_the_turn_shows_the_n_rows_asked_for(offline, monkeypatch):  # noqa: F811
    # The model calls the right report without `top`; the table and chart still hold ten rows.
    seen = {}

    def report(**kw):
        seen.update(kw)
        return {**_RANKED, "view": kw.get("view")}
    monkeypatch.setitem(kr.KPI_REGISTRY["oof-margin-loss"], "fn", report)
    monkeypatch.setattr(deep_tools, "_kpi_scope_mismatch", lambda *a, **k: None)
    offline(FakeClient(rounds=[
        ("", [("get_kpi", {"kpi": "oof-margin-loss", "month": "Aug 2026", "view": "items"})]),
        ("BRAND 0 missed the most margin.",
         [("present", {"charts": [{"type": "bar", "x": "item", "y": "margin_loss_value"}]})]),
    ]))
    events = list(orch.answer(TOP10))
    assert "top" not in seen                                     # call_kpi's own; never the handler's
    sql = next(e for e in events if e["type"] == "sql")
    assert "get_kpi('oof-margin-loss'" in sql["sql"] and "top=10" in sql["sql"]
    table = next(e for e in events if e["type"] == "table")["table"]
    assert len(table["rows"]) == 10
    chart = next(e for e in events if e["type"] == "chart")["plotly"]
    assert len(chart["data"][0]["x"]) == 10
    assert next(e for e in events if e["type"] == "answer")["verified"] == "canonical"
