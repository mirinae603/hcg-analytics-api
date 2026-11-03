# tests/test_formulary_basis.py — Basis=qty on /formulary/*, each unit's consumption split and
# its all-units Total row (unit_mix_total), and the notes worded for the basis (meta included).
#
# The fixture is tests/test_formulary_api.py's 15 invented lines, written by its own table
# writers, with one change: every line sells its own number of units (QTY below) instead of
# 2, so a units figure can differ from the value one. Every expected number was worked out by
# hand from those lines (the comments show the arithmetic). Lines 1-15 are numbered as there.
#
# The `real` tests at the end pin Aug-26 figures on the committed tables and are skipped when
# the tables (or that month) are not built.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.api import formulary
from app.core.config import settings

from tests import test_formulary_api as base

AUG, JUL = base.AUG, base.JUL
KR, DR, RAJKOT = base.KR, base.DR, base.RAJKOT

# Units sold on each of base.LINES, in its order. Line 14 (Rajkot's only line) sells 0 units,
# so Rajkot has value but no units: its unit split is null on qty, never 0 / 0 / 0 or NaN.
# Line 8 (100003) and lines 3 + 4 (200001) both come to 45 units: equal counts in one list.
QTY = [10, 30, 40, 5, 15, 20, 99, 45, 4, 6, 5, 45, 10, 0, 35]
_BASE_LINES = base._lines


def _lines_with_units() -> pd.DataFrame:
    df = _BASE_LINES()
    assert len(df) == len(QTY), "base.LINES changed: give every line its units here"
    df["qty"] = [float(q) for q in QTY]
    return df


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    monkeypatch.setattr(base, "_lines", _lines_with_units)
    base.write_tables(tmp_path / "formulary")
    base.write_hospital_map(tmp_path / "reference")
    formulary.clear_caches()
    yield tmp_path
    formulary.clear_caches()


def overview(client, **params):
    r = client.get("/formulary/overview", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def table(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def split(rows) -> dict:
    return {r["plant"]: (r["total"], r["formulary_share_pct"], r["oof_share_pct"], r["nf_share_pct"])
            for r in rows}


def approx(*v):
    return pytest.approx(tuple(v), abs=1e-4)


# ---------- the basis itself ----------

def test_qty_counts_units_with_the_same_split_and_ratios(client, data):
    d = overview(client, Plant="HC05", Month=AUG, Basis="qty")
    h = d["headline"]
    # KR Aug drugs in units: A = 10 + 40 + 5 + 45 (lines 1, 3, 4, 8), B = 30 + 15 (2, 5), C = 20 (6);
    # the consumable (line 7, 99 units) is outside.
    assert (h["formulary"], h["out_of_formulary"], h["non_formulary"]) == (100.0, 45.0, 20.0)
    assert (h["total"], h["in_ratio"], h["lines"]) == (165.0, 145.0, 7)       # lines stay a line count
    assert h["pooled_pct"] == pytest.approx(68.9655, abs=1e-4)                 # 100 / 145
    assert h["onco_pct"] == pytest.approx(64.7059, abs=1e-4)                   # (10 + 45) / (10 + 45 + 30)
    assert h["non_onco_pct"] == pytest.approx(75.0)                            # 45 / (45 + 15)
    assert h["mean_pct"] == pytest.approx(69.8529, abs=1e-4)                   # (64.7059 + 75) / 2
    assert h["compliance_pct"] == h["pooled_pct"]
    # never judged against the budget (a drug-cost target), and says so
    assert (h["budget_pct"], h["budget_source"]) == (None, None)
    assert (d["scope"]["basis"], d["scope"]["budget_judged"]) == ("qty", False)
    assert "units sold" in d["notes"][0]
    assert any(n.startswith("Quantity view") for n in d["notes"])
    # the same call on value: HCG's figures, set against the budget (TARGET_PCT: no MIS here)
    v = overview(client, Plant="HC05", Month=AUG)
    assert (v["headline"]["formulary"], v["headline"]["budget_pct"], v["headline"]["budget_source"]) == (
        1250.0, 100.0, "default")
    assert (v["scope"]["basis"], v["scope"]["budget_judged"]) == ("value", True)
    assert not any(n.startswith("Quantity view") for n in v["notes"])


def test_value_basis_is_the_default_output(client, data):
    for q in ({"Plant": "HC05", "Month": AUG}, {"Month": "ALL", "Method": "mean"},
              {"Month": AUG, "Payer": "CASH", "Segment": "onco"}, {"Plant": "HC01"}):
        assert overview(client, Basis="value", **q) == overview(client, **q), q
        assert overview(client, Basis=" VALUE ", **q) == overview(client, **q), q
    for path, q in (("/formulary/doctors", {"Month": AUG}), ("/formulary/doctors", {"Material": "100002"}),
                    ("/formulary/items", {"Group": "A", "facets": 1}), ("/formulary/items", {"Doctor": "D1"}),
                    ("/formulary/scheme/items", {"Month": "ALL"})):
        assert table(client, path, Basis="value", **q) == table(client, path, **q), (path, q)
    assert overview(client, Plant="HC05", Month=AUG, Basis="QTY")["scope"]["basis"] == "qty"


def test_a_bad_basis_is_a_400(client, data):
    for path in ("/formulary/meta", "/formulary/overview", "/formulary/doctors", "/formulary/items",
                 "/formulary/scheme/items"):
        for bad in ("units", "cost", "mrp", "lines", "1"):
            r = client.get(path, params={"Basis": bad})
            assert r.status_code == 400, (path, bad, r.text)
            assert "Basis" in r.json()["detail"], (path, bad)


def test_python_callers_get_the_value_basis(data):
    # brand_rules calls _scope positionally up to Payer; app/ai/hcg_reports calls the handler
    # as a function, where untouched Query(None) defaults are fastapi objects, not None.
    assert formulary._scope("HC05", AUG, None, None, None, None).basis == "value"
    assert formulary._scope("HC05", AUG, None, None, None, None, "mean", "qty").basis == "qty"
    d = formulary.formulary_overview(Plant="HC05", Month=AUG)
    assert (d["scope"]["basis"], d["headline"]["formulary"]) == ("value", 1250.0)


# ---------- each unit's split of its own consumption (A + B + C) ----------

def test_unit_split_adds_to_100_on_each_basis(client, data):
    # value, Aug: KR A 1250 B 250 C 200 (1700); DR A 500 B 500 + 40 C 0 (1040); Rajkot C 70 only.
    rows = overview(client, Month=AUG)["by_hospital"]
    assert split(rows) == {KR: approx(1700.0, 73.5294, 14.7059, 11.7647),
                           DR: approx(1040.0, 48.0769, 51.9231, 0.0),
                           RAJKOT: approx(70.0, 0.0, 0.0, 100.0)}
    # units, Aug: KR 100 / 45 / 20 (165); DR A 5, B 45 + 35 (85); Rajkot sold 0 units -> no split.
    rows = overview(client, Month=AUG, Basis="qty")["by_hospital"]
    assert split(rows) == {KR: approx(165.0, 60.6061, 27.2727, 12.1212),
                           DR: approx(85.0, 5.8824, 94.1176, 0.0),
                           RAJKOT: (0.0, None, None, None)}
    for basis in ("value", "qty"):
        for r in overview(client, Month="ALL", Basis=basis)["by_hospital"]:
            parts = (r["formulary_share_pct"], r["oof_share_pct"], r["nf_share_pct"])
            if r["total"]:
                assert sum(parts) == pytest.approx(100.0, abs=1e-3), (basis, r["plant"])
            else:
                assert parts == (None, None, None), (basis, r["plant"])


def test_unit_split_follows_the_ranking_filters(client, data):
    # Payer=CASH, value: KR A 300 (3) B 100 (2); DR A 500 (11) B 500 (12); Rajkot C 70 (14).
    rows = overview(client, Plant="HC05", Month=AUG, Payer="CASH")["by_hospital"]
    assert split(rows) == {KR: approx(400.0, 75.0, 25.0, 0.0), DR: approx(1000.0, 50.0, 50.0, 0.0),
                           RAJKOT: approx(70.0, 0.0, 0.0, 100.0)}
    assert [r["selected"] for r in rows if r["plant"] == KR] == [True]
    # Segment=onco, units: KR A 10 + 45 B 30; DR A 5 B 45 + 35; Rajkot sells no onco -> no row.
    rows = overview(client, Month=AUG, Segment="onco", Basis="qty")["by_hospital"]
    assert split(rows) == {DR: approx(85.0, 5.8824, 94.1176, 0.0), KR: approx(85.0, 64.7059, 35.2941, 0.0)}
    # Month=ALL, units: KR adds July's A 4 (line 9) and C 6 (line 10) -> 104 / 45 / 26 of 175.
    rows = overview(client, Month="ALL", Basis="qty")["by_hospital"]
    assert split(rows)[KR] == approx(175.0, 59.4286, 25.7143, 14.8571)


# ---------- the rest of the overview on units ----------

def test_qty_ranking_trend_network_and_no_budget(client, data):
    d = overview(client, Plant="HC05", Month=AUG, Basis="qty")
    rows = d["by_hospital"]
    assert [(r["plant"], r["rank"]) for r in rows] == [(DR, 1), (KR, 2), (RAJKOT, None)]
    assert rows[0]["compliance_pct"] == pytest.approx(5.8824, abs=1e-4)          # 5 / 85
    assert all(r["budget_pct"] is None for r in rows)
    assert all(r["budget_pct"] == 100.0 for r in overview(client, Plant="HC05", Month=AUG)["by_hospital"])
    # Aug, every unit: A 100 + 5, B 45 + 80 -> 105 / 230
    assert d["network"] == {"compliance_pct": pytest.approx(45.6522, abs=1e-4)}
    # KR July: A 4, C 6 -> 100 %; every month, whatever Month says; no budget on any month.
    assert [(t["period"], t["compliance_pct"], t["total"], t["budget_pct"]) for t in d["trend"]] == [
        (JUL, 100.0, 10.0, None), (AUG, pytest.approx(68.9655, abs=1e-4), 165.0, None)]
    seg = {s["key"]: s for s in d["segments"]}
    assert (seg["onco"]["formulary"], seg["onco"]["out_of_formulary"]) == (55.0, 30.0)
    assert {t["type"]: t["compliance_pct"] for t in d["by_type"]} == {
        "IP": pytest.approx(55 / 85 * 100, abs=1e-4),        # IP: A 10 + 45, B 30, C 20 (lines 1, 8, 2, 6)
        "OP": pytest.approx(75.0)}                           # OP: A 40 + 5, B 15 (lines 3, 4, 5)


def test_qty_class_mix_scheme_consumption_and_payer(client, data):
    d = overview(client, Plant="HC05", Month=AUG, Basis="qty")
    # by_class keeps its `cost` key; on qty it holds units. Of 165 units:
    by = {r["formulary_class"]: (r["cost"], r["share_pct"]) for r in d["by_class"]}
    assert by == {"FORMULARYP": approx(45.0, 27.2727), "FORMULARYR-OPD": approx(45.0, 27.2727),
                  "FORMULARYR": approx(10.0, 6.0606), "OUT OF FORMULARY": approx(45.0, 27.2727),
                  "NON FORMULARY": approx(20.0, 12.1212)}
    assert sum(r["share_pct"] for r in d["by_class"]) == pytest.approx(100.0, abs=1e-3)
    sc = d["scheme"]
    # restricted on GOV: line 1 (10) + line 8 (45) = 55 of 165; onco 55 of 85 (lines 1, 2, 8)
    assert (sc["restricted"]["cost"], sc["restricted"]["lines"]) == (55.0, 2)
    assert sc["restricted"]["share_pct"] == pytest.approx(33.3333, abs=1e-4)
    assert sc["restricted"]["onco"] == {"cost": 55.0, "share_pct": pytest.approx(64.7059, abs=1e-4)}
    assert sc["preferred"]["non_onco"] == {"cost": 5.0, "share_pct": pytest.approx(6.25)}  # 5 of 80
    assert sc["gov_total"] == 80.0                                                  # 10 + 5 + 20 + 45
    assert sc["base"] == {"all": 165.0, "onco": 85.0, "non_onco": 80.0}
    total = next(r for r in d["consumption"]["rows"] if r["segment"] == "all")
    assert (total["formulary"], total["out_of_formulary"], total["non_formulary"], total["total"]) == (
        100.0, 45.0, 20.0, 165.0)
    assert (total["formulary_pct"], total["oof_pct"], total["nf_pct"]) == approx(60.6061, 27.2727, 12.1212)
    assert (total["restricted_scheme"], total["preferred_scheme"]) == (55.0, 5.0)
    assert total["preferred_scheme_pct"] == pytest.approx(3.0303, abs=1e-4)        # 5 / 165
    assert "on units sold" in d["consumption"]["scope_note"]
    # payer-wise, consumables included: CASH 30 + 40 + 99, GOV 55 + 25, CASH(TPA) 15 -> 264
    pay = {r["payer"]: r for r in d["by_payer"]["rows"]}
    assert (pay["CASH"]["total"], pay["CASH"]["consumables"], pay["ALL"]["total"]) == (169.0, 99.0, 264.0)
    assert pay["CASH"]["share_pct"] == pytest.approx(64.0152, abs=1e-4)
    cells = {(c["formulary_class"], c["payer"]): c["onco_ip"] for c in d["by_payer"]["classes"]}
    assert cells[("FORMULARYR-OPD", "GOV")] == 45.0


# ---------- lists ----------

def test_doctors_on_qty_count_units_and_sort_by_them(client, data):
    body = table(client, "/formulary/doctors", Plant="HC05", Month=AUG, Basis="qty")
    # OOF units: D1 30 (line 2), D2 15 (line 5) -> D1 first; on value D2 (150) leads D1 (100).
    assert [r["doctor_id"] for r in body["data"]] == ["D1", "D2", "UNKNOWN"]
    assert [r["doctor_id"] for r in table(client, "/formulary/doctors", Plant="HC05", Month=AUG)["data"]] == [
        "D2", "D1", "UNKNOWN"]
    d1 = body["data"][0]
    # lines 1, 2, 6: A 10, B 30, C 20 -> 10 / 40; OOF share 30 of 45
    assert {k: d1[k] for k in ("formulary", "out_of_formulary", "non_formulary", "total", "onco", "non_onco",
                               "lines")} == {"formulary": 10.0, "out_of_formulary": 30.0, "non_formulary": 20.0,
                                             "total": 60.0, "onco": 40.0, "non_onco": 20.0, "lines": 3}
    assert d1["compliance_pct"] == pytest.approx(25.0) and d1["oof_share_pct"] == pytest.approx(66.6667, abs=1e-4)
    assert body["sums"]["out_of_formulary"] == 45.0


def test_doctors_behind_one_item_sort_by_units(client, data):
    body = table(client, "/formulary/doctors", Month=AUG, Material="100002", Basis="qty")
    # units 45 (line 12), 35 (line 15), 30 (line 2); on value 500, 100, 40 puts HC05 before HC06.
    assert [(r["doctor_id"], r["plant"], r["cost"], r["qty"]) for r in body["data"]] == [
        ("D4", "HC06", 500.0, 45.0), ("D1", "HC06", 40.0, 35.0), ("D1", "HC05", 100.0, 30.0)]
    assert body["sums"] == {"cost": 640.0, "qty": 110.0, "lines": 3}


def test_items_on_qty_share_and_sort_by_units_and_keep_cost(client, data):
    a = table(client, "/formulary/items", Plant="HC05", Month=AUG, Group="A", Basis="qty")
    # 200001: 40 + 5 units (₹350), 100003: 45 (₹100), 100001: 10 (₹800). Equal units -> dearer first.
    assert [(r["material"], r["cost"], r["qty"], r["share_pct"]) for r in a["data"]] == [
        ("200001", 350.0, 45.0, 45.0), ("100003", 100.0, 45.0, 45.0), ("100001", 800.0, 10.0, 10.0)]
    b = table(client, "/formulary/items", Month=AUG, Basis="qty")
    # Group B, every unit: 100002 = 30 + 45 + 35 units, 200002 = 15 -> shares of 125
    assert [(r["material"], r["qty"], r["share_pct"]) for r in b["data"]] == [
        ("100002", 110.0, 88.0), ("200002", 15.0, 12.0)]
    assert b["sums"] == {"cost": 790.0, "qty": 125.0, "lines": 4}
    one = table(client, "/formulary/items", Month=AUG, Doctor="D1", Basis="qty")["data"]
    assert [(r["material"], r["cost"], r["qty"], r["share_pct"], r["hospitals"]) for r in one] == [
        ("100002", 140.0, 65.0, 100.0, 2)]                                    # lines 2 + 15
    s = table(client, "/formulary/scheme/items", Plant="HC05", Month=AUG, Basis="qty")
    assert [(r["material"], r["cost"], r["qty"]) for r in s["data"]] == [
        ("100003", 100.0, 45.0), ("100001", 800.0, 10.0)]                     # value order is the reverse


def test_list_scope_chips_say_units(client, data):
    chips = table(client, "/formulary/items", Month=AUG, Basis="qty", facets=1)["scope"]
    basis = [c for c in chips if c["param"] == "Basis"]
    assert basis == [{"param": "Basis", "label": "Counted in", "value": "qty", "value_label": "Units",
                      "token": basis[0]["token"]}]
    assert not any(c["param"] == "Basis" for c in table(client, "/formulary/items", Month=AUG, facets=1)["scope"])


# ---------- edge cases ----------

def test_qty_edge_cases(client, data):
    # a known plant with no formulary rows: nulls, no budget, the ranking still there
    d = overview(client, Plant="HC01", Basis="qty")
    assert d["headline"]["compliance_pct"] is None and d["headline"]["lines"] == 0
    assert d["headline"]["budget_pct"] is None and len(d["by_hospital"]) == 3
    # Rajkot sells 0 units: no ratio and no split, not a divide error
    h = overview(client, Plant="HS01", Month=AUG, Basis="qty")["headline"]
    assert (h["total"], h["compliance_pct"], h["pooled_pct"], h["mean_pct"]) == (0.0, None, None, None)
    # an empty selection on the lists
    assert table(client, "/formulary/items", Plant="HC05", Month=AUG, Group="C", Segment="onco",
                 Basis="qty")["total"] == 0
    for path in ("/formulary/overview", "/formulary/doctors", "/formulary/items", "/formulary/scheme/items"):
        r = client.get(path, params={"Plant": "NOPE", "Basis": "qty"})
        assert r.status_code == 404, (path, r.text)


def test_missing_tables_are_a_503_on_every_basis(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "empty"))
    formulary.clear_caches()
    for path in ("/formulary/meta", "/formulary/overview", "/formulary/doctors", "/formulary/items",
                 "/formulary/scheme/items"):
        for basis in ("qty", "value", "nonsense"):            # no data is said first, as for every param
            r = client.get(path, params={"Basis": basis})
            assert r.status_code == 503, (path, basis, r.text)
    formulary.clear_caches()


# ---------- the unit split's Total row (unit_mix_total) ----------

def mix_total(d) -> tuple:
    u = d["unit_mix_total"]
    return (u["formulary"], u["out_of_formulary"], u["non_formulary"], u["total"],
            u["formulary_share_pct"], u["oof_share_pct"], u["nf_share_pct"])


def test_unit_mix_total_is_every_unit_together(client, data):
    # value, Aug: A 1250 (KR) + 500 (DR) = 1750; B 250 + 540 = 790; C 200 + 70 (Rajkot) = 270; of 2810
    v = overview(client, Month=AUG)
    assert mix_total(v) == approx(1750.0, 790.0, 270.0, 2810.0, 62.2776, 28.1139, 9.6085)
    # a ratio of the sums, not the mean of the units' formulary shares: (73.5294 + 48.0769 + 0) / 3
    mean = sum(r["formulary_share_pct"] for r in v["by_hospital"]) / 3
    assert mean == pytest.approx(40.5354, abs=1e-4)
    assert v["unit_mix_total"]["formulary_share_pct"] != pytest.approx(mean, abs=1)
    # units, Aug: A 100 + 5 = 105; B 45 + 80 = 125; C 20 + 0 = 20; of 250 (Rajkot's 0 units add nothing)
    q = overview(client, Month=AUG, Basis="qty")
    assert mix_total(q) == approx(105.0, 125.0, 20.0, 250.0, 42.0, 50.0, 8.0)
    for d in (v, q):
        u, rows = d["unit_mix_total"], d["by_hospital"]
        # the sum of the ranking's rows, and on all hospitals the headline's A / B / C
        for k in ("formulary", "out_of_formulary", "non_formulary", "total"):
            assert u[k] == pytest.approx(sum(r[k] for r in rows), abs=0.01), k
        assert (u["formulary"], u["out_of_formulary"], u["non_formulary"]) == (
            d["headline"]["formulary"], d["headline"]["out_of_formulary"], d["headline"]["non_formulary"])
        assert u["formulary_share_pct"] + u["oof_share_pct"] + u["nf_share_pct"] == pytest.approx(100, abs=1e-3)
        keys = list(d)
        assert keys[keys.index("network") + 1] == "unit_mix_total"          # beside the network figure
    # HCG's consumption Total row is the same split when no Payer or Segment narrows the ranking
    whole = next(r for r in q["consumption"]["rows"] if r["segment"] == "all")
    assert (whole["formulary_pct"], whole["oof_pct"], whole["nf_pct"]) == approx(42.0, 50.0, 8.0)


def test_unit_mix_total_follows_the_ranking_filters(client, data):
    # Payer=CASH, value, Aug: KR A 300 (3) B 100 (2); DR A 500 (11) B 500 (12); Rajkot C 70 (14).
    # The hospital selector is ignored, as the ranking ignores it.
    cash = approx(800.0, 600.0, 70.0, 1470.0, 54.4218, 40.8163, 4.7619)
    assert mix_total(overview(client, Month=AUG, Payer="CASH")) == cash
    assert mix_total(overview(client, Plant="HC05", Month=AUG, Payer="CASH")) == cash
    # Segment=onco, units, Aug: KR A 10 + 45 B 30; DR A 5 B 45 + 35 -> 60 / 110 / 0 of 170
    assert mix_total(overview(client, Month=AUG, Segment="onco", Basis="qty")) == approx(
        60.0, 110.0, 0.0, 170.0, 35.2941, 64.7059, 0.0)
    # Type=OP, value, Aug: KR A 300 + 50 B 150 (3, 4, 5); Rajkot C 70 (14) -> of 570
    assert mix_total(overview(client, Month=AUG, Type="OP")) == approx(
        350.0, 150.0, 70.0, 570.0, 61.4035, 26.3158, 12.2807)
    # Month=JUL, value: KR A 400 (9) C 50 (10); DR A 100 (13) -> of 550; no out of formulary at all
    assert mix_total(overview(client, Month=JUL)) == approx(500.0, 0.0, 50.0, 550.0, 90.9091, 0.0, 9.0909)
    # Month=ALL, units: Aug's 105 / 125 / 20 plus July's A 4 + 10, C 6 -> 119 / 125 / 26 of 270
    assert mix_total(overview(client, Month="ALL", Basis="qty")) == approx(
        119.0, 125.0, 26.0, 270.0, 44.0741, 46.2963, 9.6296)
    # the method changes the ratio, never the split
    assert overview(client, Month="ALL", Method="mean")["unit_mix_total"] == overview(
        client, Month="ALL")["unit_mix_total"]


def test_unit_mix_total_edge_cases(client, data):
    # nothing sold anywhere in the selection (every onco line is IP): no rows, zeros, null shares
    for basis in ("value", "qty"):
        r = client.get("/formulary/overview", params={"Month": AUG, "Type": "OP", "Segment": "onco", "Basis": basis})
        assert r.status_code == 200 and "NaN" not in r.text and "Infinity" not in r.text
        d = r.json()
        assert d["by_hospital"] == []
        assert mix_total(d) == (0.0, 0.0, 0.0, 0.0, None, None, None)
    # a hospital with no formulary rows still gets every unit's Total row (the selector is ignored)
    assert overview(client, Plant="HC01", Basis="qty")["unit_mix_total"] == overview(
        client, Basis="qty")["unit_mix_total"]


# ---------- notes worded for the basis ----------

VALUE_PAGE_NOTES = [
    "Compliance = Formulary ÷ (Formulary + Out of Formulary), on drug cost. Non Formulary is outside the ratio; "
    "consumables are excluded.",
    "Pooled: one ratio over onco and non-onco drug spend together (HCG's current method).",
    "Mean: the average of the onco ratio and the non-onco ratio (HCG's earlier method); a segment with no "
    "formulary or out-of-formulary spend is left out.",
    "Onco / Non-Onco follow the material code's first digit (1 = onco, 2 = non-onco).",
]


def test_meta_notes_follow_the_basis(client, data):
    value = table(client, "/formulary/meta")
    assert value["notes"] == VALUE_PAGE_NOTES                    # HCG's wording, as before Basis existed
    assert table(client, "/formulary/meta", Basis="value") == value
    assert table(client, "/formulary/meta", Basis=" Value ") == value
    units = table(client, "/formulary/meta", Basis="qty")
    assert units["notes"][:4] == [
        "Compliance = Formulary ÷ (Formulary + Out of Formulary), on units sold. Non Formulary is outside the "
        "ratio; consumables are excluded.",
        "Pooled: one ratio over onco and non-onco drug units sold together (HCG's current method).",
        "Mean: the average of the onco ratio and the non-onco ratio (HCG's earlier method); a segment with no "
        "formulary or out-of-formulary units sold is left out.",
        formulary.NOTE_SEGMENT]
    assert units["notes"][4] == formulary.NOTE_UNITS and len(units["notes"]) == 5
    assert not any("drug cost" in n or "spend" in n for n in units["notes"])
    # only the notes follow the basis: months, hospitals, options and sources are the same
    assert {k: v for k, v in units.items() if k != "notes"} == {k: v for k, v in value.items() if k != "notes"}


def test_overview_notes_are_the_page_notes_of_their_basis(client, data):
    # The page lists the answer's notes and meta's together, each once (it drops exact repeats):
    # so on either basis and method the answer's ratio, method and segment notes are meta's own.
    for basis in ("value", "qty"):
        page = table(client, "/formulary/meta", Basis=basis)["notes"]
        for method, i in (("pooled", 1), ("mean", 2)):
            notes = overview(client, Plant="HC05", Month=AUG, Method=method, Basis=basis)["notes"]
            assert notes[:3] == [page[0], page[i], page[3]], (basis, method)
    q = overview(client, Month="ALL", Segment="onco", Method="mean", Basis="qty")["notes"]
    assert not any("drug cost" in n or "spend" in n for n in q)
    assert q[1] == ("Mean: the average of the onco ratio and the non-onco ratio (HCG's earlier method); a segment "
                    "with no formulary or out-of-formulary units sold is left out.")


def test_the_no_data_note_says_units_on_qty(client, data):
    v = overview(client, Plant="HC01")
    q = overview(client, Plant="HC01", Basis="qty")
    name = v["scope"]["plant_name"]
    assert v["notes"][-1] == f"No formulary drug spend for {name} in this selection."
    assert q["notes"][-1] == f"No formulary drug units sold for {name} in this selection."
    assert not any("spend" in n for n in q["notes"])
    # a hospital with lines gets no such note on either basis
    for basis in ("value", "qty"):
        assert not any(n.startswith("No formulary drug") for n in overview(
            client, Plant="HC05", Month=AUG, Basis=basis)["notes"])


# ---------- real tables: Aug-26 ----------

_SUMMARY = Path(settings.FORMULARY_DATA_DIR) / "formulary_summary.parquet"
real = pytest.mark.skipif(not _SUMMARY.exists(),
                          reason="formulary tables not built — run python -m app.etl.formulary")
REAL_MONTH = "2026-08"


def _need_real_month():
    if REAL_MONTH not in set(pd.read_parquet(_SUMMARY, columns=["period"])["period"].astype(str)):
        pytest.skip(f"not built: {REAL_MONTH}")


@real
def test_real_august_network_on_units(client):
    _need_real_month()
    d = overview(client, Month=REAL_MONTH, Basis="qty")
    h = d["headline"]
    # Aug-26 drugs, every unit, in units: pooled 78.61 % (97.55 % on value)
    assert h["pooled_pct"] == pytest.approx(78.6082, abs=1e-4)
    assert d["network"]["compliance_pct"] == h["compliance_pct"] == h["pooled_pct"]
    whole = next(r for r in d["consumption"]["rows"] if r["segment"] == "all")
    assert (whole["formulary_pct"], whole["oof_pct"], whole["nf_pct"]) == approx(58.8834, 16.024, 25.0925)
    units = d["by_hospital"]
    assert sum(r["total"] for r in units) == pytest.approx(
        h["formulary"] + h["out_of_formulary"] + h["non_formulary"], abs=0.5)
    assert all(r["budget_pct"] is None for r in units) and d["scope"]["budget_judged"] is False
    for r in units:
        assert r["formulary_share_pct"] + r["oof_share_pct"] + r["nf_share_pct"] == pytest.approx(100, abs=1e-3)


@real
def test_real_august_unit_split_on_value(client):
    _need_real_month()
    d = overview(client, Month=REAL_MONTH)
    assert d["headline"]["pooled_pct"] == pytest.approx(97.5547, abs=1e-4)      # as test_formulary_golden
    units = d["by_hospital"]
    assert len(units) == 23
    f = [r["formulary_share_pct"] for r in units]
    assert (min(f), max(f)) == approx(56.5645, 100.0)
    assert max(r["oof_share_pct"] for r in units) == pytest.approx(17.0813, abs=1e-4)
    assert max(r["nf_share_pct"] for r in units) == pytest.approx(28.446, abs=1e-4)
    assert min(min(r["oof_share_pct"], r["nf_share_pct"]) for r in units) == 0.0
    for r in units:
        assert r["formulary_share_pct"] + r["oof_share_pct"] + r["nf_share_pct"] == pytest.approx(100, abs=1e-3)
    assert any(r["budget_pct"] is not None for r in units) and d["scope"]["budget_judged"] is True


@real
def test_real_august_unit_mix_total(client):
    _need_real_month()
    v = overview(client, Month=REAL_MONTH)
    q = overview(client, Month=REAL_MONTH, Basis="qty")
    # Aug-26, all 23 units: 91.43 / 2.29 / 6.27 % of ₹46.37 Cr of drug cost; on units 58.88 / 16.02 / 25.09 %
    assert mix_total(v)[3:] == approx(463658893.74, 91.4344, 2.2919, 6.2736)
    assert mix_total(q)[3:] == approx(3243745.0, 58.8834, 16.024, 25.0925)
    for d in (v, q):
        u, rows = d["unit_mix_total"], d["by_hospital"]
        for k in ("formulary", "out_of_formulary", "non_formulary", "total"):
            assert u[k] == pytest.approx(sum(r[k] for r in rows), abs=0.01 * len(rows)), k
        assert (u["formulary"], u["out_of_formulary"], u["non_formulary"]) == (
            d["headline"]["formulary"], d["headline"]["out_of_formulary"], d["headline"]["non_formulary"])
        whole = next(r for r in d["consumption"]["rows"] if r["segment"] == "all")
        assert (u["formulary_share_pct"], u["oof_share_pct"], u["nf_share_pct"]) == (
            whole["formulary_pct"], whole["oof_pct"], whole["nf_pct"])
    # a hospital selected changes nothing: the Total row is every unit's, as the ranking is
    assert overview(client, Plant="HC05", Month=REAL_MONTH, Basis="qty")["unit_mix_total"] == q["unit_mix_total"]
