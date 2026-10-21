# tests/test_email_coverage_formulary.py — what HCG's two monthly emails show, on the formulary
# pages: totals under every table (paginate sums), HCG's onco / non-onco consumption table, the
# payer-wise consumption and S5 'Summary' layout, and the bases behind each %.
#
# The hand-worked tests use tests/test_formulary_api.py's 15 invented lines (line numbers below
# are positions in its LINES, 1-based) and test_brand_rules_api's rule fixture. The real-data
# tests at the end reproduce HCG's KR Aug-26 email figures and skip when the tables are absent.
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest
from starlette.requests import Request

from app.api import brand_rules, formulary
from app.core import data_access as da
from app.core.config import settings
from tests.test_brand_rules_api import rules_data  # noqa: F401  (fixture)
from tests.test_formulary_api import AUG, JUL, data  # noqa: F401  (fixture)


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def rows_by(rows, key):
    return {r[key]: r for r in rows}


# ---------- Contract A: paginate sums (no client: a plain frame) ----------

def test_paginate_sums_cover_the_filtered_rows_not_the_page():
    df = pd.DataFrame({"name": ["alpha", "beta", "gamma", "alphabet"], "cost": [0.1, 0.2, 10.0, 5.0],
                       "lines": pd.Series([1, 2, 3, 4], dtype="int64"), "blank": [None, None, None, None],
                       "hidden": [1.0, 1.0, 1.0, 1.0]})
    legacy = da.paginate("x", None, {"page_size": 1}, col_map={}, frame=df)
    assert set(legacy) == {"data", "total"}                          # no sum_cols: response unchanged
    out = da.paginate("x", None, {"page_size": 1, "global_filter": "alpha"}, col_map={}, frame=df,
                      sum_cols=["cost", "lines", "blank", "missing", "hidden"],
                      columns=["name", "cost", "lines", "blank"], rename={"cost": "spend"})
    assert out["total"] == 2 and len(out["data"]) == 1
    # alpha + alphabet: 0.1 + 5.0 (search applied, paging not); renamed key; count stays an int;
    # an all-null column is None; a column the table does not return is skipped.
    assert out["sums"] == {"spend": 5.1, "lines": 5, "blank": None}
    assert isinstance(out["sums"]["lines"], int)
    noise = da.paginate("x", None, {}, col_map={}, frame=df.iloc[:2], sum_cols=["cost"])    # alpha + beta
    assert noise["sums"] == {"cost": 0.3}                            # not 0.30000000000000004
    empty = da.paginate("x", None, {"global_filter": "zzz"}, col_map={}, frame=df, sum_cols=["cost", "lines"])
    assert empty["total"] == 0 and empty["sums"] == {"cost": 0.0, "lines": 0}


def test_every_formulary_list_returns_its_totals(client, data):
    d = get(client, "/formulary/doctors", Plant="HC05", Month=AUG, page_size=1)
    # KR Aug drugs: A 1250, B 250, C 200; onco 1000 (lines 1, 2, 8), non-onco 700 (3, 4, 5, 6)
    assert d["sums"] == {"formulary": 1250.0, "out_of_formulary": 250.0, "non_formulary": 200.0,
                         "total": 1700.0, "onco": 1000.0, "non_onco": 700.0, "lines": 7}
    rows = rows_by(get(client, "/formulary/doctors", Plant="HC05", Month=AUG)["data"], "doctor_id")
    assert [(r["total"], r["onco"], r["non_onco"]) for r in (rows["D2"], rows["UNKNOWN"])] == \
        [(500.0, 0.0, 500.0), (100.0, 100.0, 0.0)]                  # lines 3+4+5; line 8
    found = get(client, "/formulary/doctors", Plant="HC05", Month=AUG, global_filter="bhanu")
    assert found["sums"]["total"] == 500.0 and found["sums"]["out_of_formulary"] == 150.0
    assert get(client, "/formulary/items", Month=AUG, page_size=1)["sums"] == \
        {"cost": 790.0, "qty": 8.0, "lines": 4}                      # B: lines 2, 5, 12, 15
    assert get(client, "/formulary/items", Plant="HC05", Month=AUG, Doctor="D1", Group="C")["sums"]["cost"] == 200.0
    assert get(client, "/formulary/doctors", Month=AUG, Material="100002")["sums"] == \
        {"cost": 640.0, "qty": 6.0, "lines": 3}                      # 500 + 100 + 40
    assert get(client, "/formulary/scheme/items", Plant="HC05", Month=AUG)["sums"] == \
        {"cost": 900.0, "qty": 4.0, "lines": 2}                      # GOV R 800 + R-OPD 100


# ---------- Contract C: overview additions ----------

def test_headline_and_trend_carry_the_amounts_behind_the_percent(client, data):
    d = get(client, "/formulary/overview", Plant="HC05", Month=AUG)
    h = d["headline"]
    assert (h["in_ratio"], h["total"], h["budget_pct"]) == (1500.0, 1700.0, 100)     # A + B; A + B + C
    t = rows_by(d["trend"], "period")
    assert {k: t[JUL][k] for k in ("formulary", "out_of_formulary", "non_formulary", "in_ratio", "total",
                                   "budget_pct")} == {"formulary": 400.0, "out_of_formulary": 0.0,
                                                      "non_formulary": 50.0, "in_ratio": 400.0,
                                                      "total": 450.0, "budget_pct": 100}   # lines 9, 10
    assert (t[AUG]["in_ratio"], t[AUG]["total"]) == (1500.0, 1700.0)
    assert get(client, "/formulary/meta")["target_pct"] == 100


def test_consumption_table_divides_by_each_segments_total(client, data):
    c = get(client, "/formulary/overview", Plant="HC05", Month=AUG)["consumption"]
    assert "non formulary IS in this denominator" in c["scope_note"]
    seg = rows_by(c["rows"], "segment")
    assert [r["segment"] for r in c["rows"]] == ["onco", "non-onco", "all"]
    onco, non, both = seg["onco"], seg["non-onco"], seg["all"]
    # onco: F 800 + 100, B 100, C 0 = 1000; GOV R + R-OPD 900
    assert (onco["formulary"], onco["out_of_formulary"], onco["non_formulary"], onco["total"]) == \
        (900.0, 100.0, 0.0, 1000.0)
    assert (onco["formulary_pct"], onco["oof_pct"], onco["nf_pct"]) == (90.0, 10.0, 0.0)
    assert (onco["restricted_scheme"], onco["restricted_scheme_pct"]) == (900.0, 90.0)
    # non-onco: F 300 + 50, B 150, C 200 = 700; GOV P 50 (line 4, not cash line 3)
    assert non["total"] == 700.0 and non["formulary_pct"] == 50.0
    assert non["oof_pct"] == pytest.approx(150 / 7, abs=1e-4) and non["nf_pct"] == pytest.approx(200 / 7, abs=1e-4)
    assert (non["preferred_scheme"], non["preferred_scheme_pct"]) == (50.0, pytest.approx(50 / 7, abs=1e-4))
    assert (both["total"], both["restricted_scheme"], both["preferred_scheme"]) == (1700.0, 900.0, 50.0)
    assert both["formulary_pct"] == pytest.approx(1250 / 17, abs=1e-4)     # not the 83.33% compliance
    # Payer and Segment do not cut HCG's table; IP/OP does.
    assert get(client, "/formulary/overview", Plant="HC05", Month=AUG, Payer="CASH",
               Segment="onco")["consumption"] == c
    ip = rows_by(get(client, "/formulary/overview", Plant="HC05", Month=AUG, Type="IP")["consumption"]["rows"],
                 "segment")
    assert (ip["non-onco"]["total"], ip["non-onco"]["nf_pct"]) == (200.0, 100.0)            # line 6 only
    # restricted for cash & TPA: KR Jul line 9 (cash R 400) is all of July's onco spend
    jul = rows_by(get(client, "/formulary/overview", Plant="HC05", Month=JUL)["consumption"]["rows"], "segment")
    assert (jul["onco"]["restricted_cash_tpa"], jul["onco"]["restricted_cash_tpa_pct"]) == (400.0, 100.0)
    assert jul["non-onco"]["formulary_pct"] == 0.0 and jul["non-onco"]["nf_pct"] == 100.0  # line 10


def test_consumption_by_payer_includes_consumables_and_ignores_the_payer_filter(client, data):
    d = get(client, "/formulary/overview", Plant="HC05", Month=AUG, Payer="GOV")["by_payer"]
    rows = rows_by(d["rows"], "payer")
    assert [r["payer"] for r in d["rows"]] == ["CASH", "CASH(TPA)", "GOV", "ALL"]
    cash = rows["CASH"]                       # lines 2 (onco IP B 100), 3 (non-onco OP A 300), 7 (consumable 999)
    assert {k: cash[k] for k in ("onco", "non_onco", "consumables", "drugs", "total", "onco_ip", "onco_op",
                                 "non_onco_ip", "non_onco_op", "formulary", "out_of_formulary",
                                 "non_formulary")} == {
        "onco": 100.0, "non_onco": 300.0, "consumables": 999.0, "drugs": 400.0, "total": 1399.0,
        "onco_ip": 100.0, "onco_op": 0.0, "non_onco_ip": 0.0, "non_onco_op": 300.0,
        "formulary": 300.0, "out_of_formulary": 100.0, "non_formulary": 0.0}
    assert cash["compliance_pct"] == 75.0 and cash["share_pct"] == pytest.approx(1399 / 2699 * 100, abs=1e-4)
    assert rows["CASH(TPA)"]["compliance_pct"] == 0.0 and rows["CASH(TPA)"]["total"] == 150.0     # line 5
    gov = rows["GOV"]                         # lines 1, 8 (onco IP A), 4 (non-onco OP A), 6 (non-onco IP C)
    assert (gov["onco_ip"], gov["non_onco_ip"], gov["non_onco_op"], gov["total"]) == (900.0, 200.0, 50.0, 1150.0)
    assert gov["compliance_pct"] == 100.0
    every = rows["ALL"]
    assert (every["drugs"], every["consumables"], every["total"], every["share_pct"]) == (1700.0, 999.0, 2699.0, 100.0)
    assert every["compliance_pct"] == pytest.approx(1250 / 15, abs=1e-4)
    # HCG's S5 'Summary' layout: drug cost by class x payer, Excel's order (class, then payer)
    assert [(c["formulary_class"], c["payer"], c["onco_ip"], c["onco_op"], c["non_onco_ip"], c["non_onco_op"],
             c["total"]) for c in d["classes"]] == [
        ("FORMULARYP", "CASH", 0.0, 0.0, 0.0, 300.0, 300.0),
        ("FORMULARYP", "GOV", 0.0, 0.0, 0.0, 50.0, 50.0),
        ("FORMULARYR", "GOV", 800.0, 0.0, 0.0, 0.0, 800.0),
        ("FORMULARYR-OPD", "GOV", 100.0, 0.0, 0.0, 0.0, 100.0),
        ("NON FORMULARY", "GOV", 0.0, 0.0, 200.0, 0.0, 200.0),
        ("OUT OF FORMULARY", "CASH", 100.0, 0.0, 0.0, 0.0, 100.0),
        ("OUT OF FORMULARY", "CASH(TPA)", 0.0, 0.0, 0.0, 150.0, 150.0)]
    assert sum(c["total"] for c in d["classes"]) == 1700.0           # = drug spend, no consumables
    assert {c["formulary_class"]: c["formulary_group"] for c in d["classes"]}["NON FORMULARY"] == "C"


def test_scheme_shares_come_with_their_base(client, data):
    sc = get(client, "/formulary/overview", Plant="HC05", Month=AUG, Payer="CASH")["scheme"]
    assert sc["base"] == {"all": 1700.0, "onco": 1000.0, "non_onco": 700.0}
    assert sc["restricted"]["share_pct"] == pytest.approx(sc["restricted"]["cost"] / sc["base"]["all"] * 100, abs=1e-4)


def test_only_computed_figures_are_served(client, data):
    # Nothing typed from HCG's emails rides along: every figure comes from the built tables.
    assert "hcg_reported" not in get(client, "/formulary/overview", Plant="HC05", Month=AUG)
    assert "reported_plants" not in get(client, "/formulary/meta")


def test_method_labels_and_notes_carry_no_fixed_months(client, data):
    m = get(client, "/formulary/meta")
    text = " ".join([x["label"] for x in m["methods"]] + m["notes"])
    assert not re.search(r"[A-Z][a-z]{2}-\d\d|FY \d\d", text)
    assert "current method" in m["methods"][0]["label"]


# ---------- brand rules ----------

def test_brand_rules_bases_and_totals(client, rules_data):
    d = get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG)
    assert [(t["label"], t["cost"], t["drug_spend"]) for t in d["trend"]] == \
        [("Jul-26", 0.0, 450.0), ("Aug-26", 900.0, 1700.0)]           # KR drugs: Jul 400 + 50, Aug 1700
    h = get(client, "/formulary/rules/overview", Plant="HC05", Month=AUG, Measure="oof_scheme")["by_hospital"]
    assert [(x["plant"], x["cost"], x["drug_spend"], x["lines"]) for x in h] == [("HC06", 40.0, 1040.0, 1)]
    assert get(client, "/formulary/rules/items", Plant="HC05", Month=AUG, page_size=1)["sums"] == \
        {"cost": 900.0, "qty": 4.0, "lines": 2}
    docs = get(client, "/formulary/rules/doctors", Plant="HC05", Month=AUG, Measure="restricted_scheme")
    assert docs["sums"] == {"cost": 900.0, "qty": 4.0, "lines": 2}  # D1 800 + UNKNOWN 100; no `brands`
    assert "sums" not in get(client, "/formulary/rules/book", Month=AUG)    # no amounts to total


# ---------- real data: HCG's KR Aug-26 email figures ----------

_SUMMARY = Path(settings.FORMULARY_DATA_DIR) / "formulary_summary.parquet"
real = pytest.mark.skipif(not _SUMMARY.exists(), reason="formulary tables not built — run python -m app.etl.formulary")


def _need_aug():
    if AUG not in set(pd.read_parquet(_SUMMARY, columns=["period"])["period"].astype(str)):
        pytest.skip("Aug-26 not built")


def _req(**q) -> Request:
    return Request({"type": "http", "query_string": urlencode(q).encode(), "headers": []})


@real
def test_kr_aug_consumption_table_matches_hcgs_spend_email():
    _need_aug()
    d = formulary.formulary_overview(Plant="HC05", Month=AUG)
    onco = rows_by(d["consumption"]["rows"], "segment")["onco"]
    assert (onco["formulary"], onco["out_of_formulary"], onco["non_formulary"]) == \
        (pytest.approx(89_803_132.00, abs=0.01), pytest.approx(890_849.24, abs=0.01),
         pytest.approx(1_303_075.32, abs=0.01))
    assert round(onco["formulary_pct"], 2) == 97.62
    assert onco["restricted_scheme"] == pytest.approx(4_215_314.63, abs=0.01)
    assert round(onco["restricted_scheme_pct"], 2) == 4.58
    assert onco["restricted_cash_tpa"] == pytest.approx(40_837_964.65, abs=0.01)
    assert round(onco["restricted_cash_tpa_pct"], 2) == 44.39
    non = rows_by(d["consumption"]["rows"], "segment")["non-onco"]
    assert (round(non["formulary_pct"], 2), round(non["oof_pct"], 2), round(non["nf_pct"], 2)) == (87.13, 5.32, 7.55)
    payer = rows_by(d["by_payer"]["rows"], "payer")
    assert payer["CASH"]["total"] == pytest.approx(42_515_900.84, abs=0.01)
    assert payer["CASH(TPA)"]["total"] == pytest.approx(69_442_051.22, abs=0.01)
    assert payer["GOV"]["total"] == pytest.approx(12_031_007.45, abs=0.01)
    assert sum(c["total"] for c in d["by_payer"]["classes"]) == pytest.approx(106_065_428.05, abs=0.05)  # S5 grand total
    h = d["headline"]
    assert h["in_ratio"] == pytest.approx(102_061_364.55 + 1_639_445.05, abs=0.01)
    assert d["scheme"]["base"]["onco"] == pytest.approx(onco["total"], abs=0.01)
    doctors = formulary.formulary_doctors(_req(page=0, page_size=1), Plant="HC05", Month=AUG)
    assert doctors["sums"]["out_of_formulary"] == pytest.approx(1_639_445.05, abs=0.01)   # ₹16.39 L
    assert doctors["sums"]["total"] == pytest.approx(h["total"], abs=0.05)
