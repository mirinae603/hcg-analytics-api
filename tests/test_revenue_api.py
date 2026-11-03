# tests/test_revenue_api.py — /revenue/overview and /revenue/materials (HCG's revenue & margin
# ask), on a tiny hand-made sales_items table.
#
# The fixture is 13 invented sales_items rows (already aggregated, as `python -m
# app.etl.formulary` publishes them) and the formulary_summary the build would derive from
# them. Every expected number was worked out by hand from those rows (comments show the
# arithmetic), not by re-running the API's code. The last tests run on the COMMITTED table and
# pin the Aug-26 network figures, skipped when it has not been built.
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.api import formulary
from app.core import data_access as da
from app.core.config import settings

from tests.test_formulary_api import write_hospital_map

JUN, JUL, AUG, MAY = "2026-06", "2026-07", "2026-08", "2026-05"
KR, DR, RAJKOT = "HC05", "HC06", "HS01"
HIS = {KR: ("KABHK", "HCG KR, Bangalore"), DR: ("KABHD", "HCG DR, Bangalore"),
       RAJKOT: ("GJHHR", "HCG Hospitals, Rajkot")}
CLASS_GROUP = {"FORMULARYR": "A", "FORMULARYP": "A", "OUT OF FORMULARY": "B",
               "NON FORMULARY": "C", "NOT IN MASTER": "X"}
CATEGORY = {"1": "Onco Drugs", "2": "Other Drugs", "3": "Consumables"}
ITEMS = {"100001": ("ONCO A 100MG", "Onco generic A", "Maker 1"),
         "100002": ("ONCO B 50MG", "Onco generic B", "Maker 2"),
         "100009": ("ONCO NEW INJ", "Onco generic N", "Maker 1"),
         "200001": ("PARA 500", "Paracetamol", "Maker 3"),
         "200003": ("VIT D3", "Cholecalciferol", "Maker 5"),
         "300001": ("SYRINGE 5ML", "Syringe", "Maker 6")}

# period, plant, type, payer, material, class, mrp, cost, qty, lines, blank-cost lines, their MRP
ROWS = [
    (AUG, KR, "IP", "GOV", "100001", "FORMULARYR", 1000, 600, 10, 4, 0, 0),           # 1
    (AUG, KR, "IP", "CASH", "100002", "OUT OF FORMULARY", 200, 150, 2, 2, 0, 0),      # 2
    (AUG, KR, "OP", "CASH", "200001", "FORMULARYP", 300, 200, 30, 3, 0, 0),           # 3
    (AUG, KR, "OP", "CASH(TPA)", "200003", "NON FORMULARY", 100, 0, 5, 1, 1, 100),    # 4 blank cost
    (AUG, KR, "IP", "CASH", "300001", "NOT IN MASTER", 400, 100, 40, 4, 1, 50),       # 5 consumable
    (AUG, KR, "IP", "GOV", "100009", "NOT IN MASTER", 50, 40, 1, 1, 0, 0),            # 6 drug not in master
    (AUG, DR, "IP", "CASH", "100001", "FORMULARYR", 3000, 2000, 20, 5, 0, 0),         # 7
    (AUG, RAJKOT, "OP", "CASH", "200003", "NON FORMULARY", 0, 0, 1, 1, 0, 0),         # 8 free: no MRP
    (JUL, KR, "IP", "CASH", "100001", "FORMULARYR", 800, 500, 8, 2, 0, 0),            # 9
    (JUL, KR, "OP", "CASH", "300001", "NOT IN MASTER", 100, 60, 10, 1, 0, 0),         # 10
    (JUL, KR, "OP", "CASH", "200001", "NON FORMULARY", 60, 30, 6, 1, 0, 0),           # 11 Jul master
    (JUL, DR, "IP", "GOV", "100002", "OUT OF FORMULARY", 500, 450, 5, 1, 0, 0),       # 12
    (JUN, DR, "IP", "CASH", "100001", "FORMULARYR", 700, 400, 7, 1, 0, 0),            # 13
]
SUMMARY_KEY = ["period", "plant", "hospital", "hospital_name", "patient_type", "category", "payer_bucket",
               "formulary_class", "formulary_group"]


def sales_items_frame(rows=ROWS) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["period", "plant", "patient_type", "payer_bucket", "material",
                                     "formulary_class", "mrp", "cost", "qty", "lines", "cost_missing_lines",
                                     "mrp_cost_missing"])
    df["hospital"] = df["plant"].map(lambda p: HIS[p][0])
    df["hospital_name"] = df["plant"].map(lambda p: HIS[p][1])
    df["material_desc"] = df["material"].map(lambda m: ITEMS[m][0])
    # July's master spelt the paracetamol differently: the newest name wins in the item list
    df.loc[(df["period"] == JUL) & (df["material"] == "200001"), "material_desc"] = "PARA 500 OLD"
    df["generic_name"] = df["material"].map(lambda m: ITEMS[m][1])
    df["manufacturer"] = df["material"].map(lambda m: ITEMS[m][2])
    df["category"] = df["material"].str[0].map(CATEGORY)
    df["formulary_group"] = df["formulary_class"].map(CLASS_GROUP)
    df = df.astype({"mrp": float, "cost": float, "qty": float, "mrp_cost_missing": float,
                    "lines": "int64", "cost_missing_lines": "int64"})
    return df[["period", "plant", "hospital", "hospital_name", "patient_type", "payer_bucket", "material",
               "material_desc", "generic_name", "manufacturer", "category", "formulary_class",
               "formulary_group", "cost", "mrp", "qty", "lines", "cost_missing_lines", "mrp_cost_missing"]]


def write_tables(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    si = sales_items_frame()
    si.to_parquet(folder / "sales_items.parquet", index=False)
    summary = si.groupby(SUMMARY_KEY, as_index=False)[["cost", "mrp", "qty", "lines"]].sum()
    # a month the summary holds and sales_items does not: revenue offers only its own months
    may = summary[summary["period"] == JUN].assign(period=MAY)
    pd.concat([summary, may]).to_parquet(folder / "formulary_summary.parquet", index=False)
    (folder / "formulary_validation.json").write_text(json.dumps({"built_at": "2026-10-30T12:00:00+00:00"}))


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "formulary")
    write_hospital_map(tmp_path / "reference")
    formulary.clear_caches()
    yield tmp_path
    formulary.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def overview(client, **params):
    return get(client, "/revenue/overview", **params)


def block(mrp, cost, pct, qty, lines, missing=0, missing_mrp=0.0):
    return {"mrp": float(mrp), "cost": float(cost), "margin": float(mrp - cost), "margin_pct": pct,
            "qty": float(qty), "lines": lines, "cost_missing_lines": missing, "mrp_cost_missing": float(missing_mrp)}


ZERO = block(0, 0, None, 0, 0)


def figures(x: dict) -> dict:
    return {k: x[k] for k in ZERO}


# ---------- headline, previous month, IP / OP ----------

def test_headline_is_mrp_less_cost_as_a_ratio_of_sums(client, data):
    d = overview(client, Plant=KR, Month=AUG)
    # rows 1-6: MRP 1000 + 200 + 300 + 100 + 400 + 50 = 2050; cost 600 + 150 + 200 + 0 + 100 + 40 = 1090;
    # margin 960 = 46.83% of MRP; qty 88; 15 lines, 2 of them with a blank cost (MRP 100 + 50)
    assert d["headline"] == block(2050, 1090, 46.83, 88, 15, 2, 150)
    assert d["scope"] == {"plant": KR, "plant_name": d["scope"]["plant_name"], "his_name": "HCG KR, Bangalore",
                          "months": [AUG], "type": "ALL", "segment": "ALL", "payer": "ALL"}
    assert d["months_available"] == [JUN, JUL, AUG]                       # not the summary's May
    assert d["month_labels"] == {JUN: "Jun-26", JUL: "Jul-26", AUG: "Aug-26"}
    # previous published month, same scope: rows 9-11, MRP 960, cost 590 -> 370 = 38.54%
    assert d["prev_period"] == JUL and d["prev"] == block(960, 590, 38.54, 24, 4)
    # IP rows 1, 2, 5, 6: 1650 / 890 -> 760 = 46.06%, 80.49% of MRP; OP rows 3, 4: 400 / 200, 19.51%
    ip, op = d["by_type"]
    assert (ip["type"], ip["label"], op["type"], op["label"]) == ("IP", "Inpatient (IP)", "OP", "Outpatient (OP)")
    assert figures(ip) == block(1650, 890, 46.06, 53, 11, 1, 50) and ip["share_pct"] == 80.49
    assert figures(op) == block(400, 200, 50.0, 35, 4, 1, 100) and op["share_pct"] == 19.51
    # no list of what HCG is to send, and no placeholders for views not built
    assert not {"awaiting", "budget", "hospital_revenue"} & set(d)
    assert d["data_as_of"] == "2026-10-30T12:00:00+00:00"
    notes = " ".join(d["notes"])
    assert "margin ÷ MRP" in notes and "GST" in notes
    assert "confirm" not in notes and "HCG has" not in notes and "HCG sends" not in notes
    assert "2 lines in this selection have a blank cost price" in notes and "₹150.00" in notes


def test_buckets_and_segments_add_up_to_the_headline(client, data):
    d = overview(client, Plant=KR, Month=AUG)
    b = {x["key"]: x for x in d["by_bucket"]}
    assert [x["key"] for x in d["by_bucket"]] == ["formulary", "out_of_formulary", "non_formulary", "consumables",
                                                  "not_in_master"]
    assert [x["label"] for x in d["by_bucket"]] == ["Formulary", "Out of formulary", "Non-formulary",
                                                    "Consumables", "Not in Item Master"]
    # formulary rows 1 + 3: 1300 / 800 -> 38.46%, 63.41% of 2050; IP row 1, OP row 3
    assert figures(b["formulary"]) == block(1300, 800, 38.46, 40, 7) and b["formulary"]["share_pct"] == 63.41
    assert b["formulary"]["ip"] == block(1000, 600, 40.0, 10, 4)
    assert b["formulary"]["op"] == block(300, 200, 33.33, 30, 3)
    assert figures(b["out_of_formulary"]) == block(200, 150, 25.0, 2, 2) and b["out_of_formulary"]["share_pct"] == 9.76
    assert b["out_of_formulary"]["op"] == ZERO
    # the blank-cost line: cost 0, so all of its MRP is margin, and it says so
    assert figures(b["non_formulary"]) == block(100, 0, 100.0, 5, 1, 1, 100) and b["non_formulary"]["share_pct"] == 4.88
    # consumables are their own bucket although their class is NOT IN MASTER (row 5)
    assert figures(b["consumables"]) == block(400, 100, 75.0, 40, 4, 1, 50) and b["consumables"]["share_pct"] == 19.51
    assert figures(b["not_in_master"]) == block(50, 40, 20.0, 1, 1) and b["not_in_master"]["share_pct"] == 2.44
    assert sum(x["mrp"] for x in d["by_bucket"]) == d["headline"]["mrp"]
    assert "Not in Item Master: drugs the month's Item Master does not list." in d["notes"]
    c = {x["key"]: x for x in d["by_category"]}
    assert [(x["key"], x["label"]) for x in d["by_category"]] == [("onco", "Onco"), ("non-onco", "Non-Onco"),
                                                                  ("consumables", "Consumables")]
    # onco rows 1, 2, 6: 1250 / 790 -> 36.8%, 60.98% of MRP
    assert figures(c["onco"]) == block(1250, 790, 36.8, 13, 7) and c["onco"]["share_pct"] == 60.98
    assert figures(c["non-onco"]) == block(400, 200, 50.0, 35, 4, 1, 100) and c["non-onco"]["share_pct"] == 19.51
    assert figures(c["consumables"]) == figures(b["consumables"])


def test_units_trend_and_the_previous_month_edges(client, data):
    d = overview(client, Month=AUG)
    # network rows 1-8: 5050 / 3090 -> 1960 = 38.81%, qty 88 + 20 + 1, lines 15 + 5 + 1;
    # July rows 9-12: 1460 / 1040 -> 420 = 28.77%
    assert d["headline"] == block(5050, 3090, 38.81, 109, 21, 2, 150)
    assert d["scope"]["plant"] == "ALL" and d["scope"]["his_name"] is None
    assert d["prev"] == block(1460, 1040, 28.77, 29, 5)
    # every unit with August sales, by MRP: DR 3000 (59.41%), KR 2050 (40.59%), Rajkot 0 (no MRP: no rank, last)
    units = d["by_hospital"]
    assert [(u["plant"], u["rank"], u["share_pct"]) for u in units] == [(DR, 1, 59.41), (KR, 2, 40.59),
                                                                        (RAJKOT, None, 0.0)]
    assert [u["his_name"] for u in units] == ["HCG DR, Bangalore", "HCG KR, Bangalore", "HCG Hospitals, Rajkot"]
    # DR's August is one IP row: 3000 - 2000 = 1000 of margin, 33.33% of MRP
    assert units[0]["his_code"] == "KABHD" and units[0]["ip"] == block(3000, 2000, 33.33, 20, 5)
    assert units[0]["op"] == ZERO and not any(u["selected"] for u in units)
    assert figures(units[2]) == block(0, 0, None, 1, 1)                 # MRP 0: margin % is null, not 0
    # a unit selected: the ranking keeps every unit and marks it
    k = overview(client, Plant="KABHK", Month=AUG)
    assert k["scope"]["plant"] == KR
    assert [(u["plant"], u["selected"]) for u in k["by_hospital"]] == [(DR, False), (KR, True), (RAJKOT, False)]
    # trend ignores Month: KR has no June sales (a zero month), July 960 / 590, August 2050 / 1090
    t = k["trend"]
    assert [(x["period"], x["label"]) for x in t] == [(JUN, "Jun-26"), (JUL, "Jul-26"), (AUG, "Aug-26")]
    assert figures(t[0]) == ZERO and t[0]["ip"] == ZERO and t[0]["op"] == ZERO
    assert figures(t[1]) == block(960, 590, 38.54, 24, 4)
    assert t[1]["ip"] == block(800, 500, 37.5, 8, 2) and t[1]["op"] == block(160, 90, 43.75, 16, 2)
    assert figures(t[2]) == k["headline"]
    # July's previous month is June, published, but KR sold nothing then: a zero block, not null
    j = overview(client, Plant=KR, Month=JUL)
    assert j["prev_period"] == JUN and j["prev"] == ZERO
    # June is the first month: no previous month, and the note says why
    f = overview(client, Plant=KR, Month=JUN)
    assert f["prev_period"] is None and f["prev"] is None
    assert f["headline"] == ZERO and f["by_bucket"][0]["share_pct"] is None
    assert [x["share_pct"] for x in f["by_type"]] == [None, None]
    assert any("first published month" in n for n in f["notes"])
    assert any(n.startswith("No pharmacy sales for") for n in f["notes"])
    # only DR sold in June, so KR (selected, no sales) is not in the ranking
    assert [(u["plant"], u["share_pct"], u["rank"]) for u in f["by_hospital"]] == [(DR, 100.0, 1)]


def test_each_unit_carries_its_own_buckets(client, data):
    d = overview(client, Month=AUG)
    units = {u["plant"]: u for u in d["by_hospital"]}
    keys = ["formulary", "out_of_formulary", "non_formulary", "consumables", "not_in_master"]
    # every unit lists the same buckets in C1's order: Not in Item Master because KR has row 6
    assert all([b["key"] for b in u["buckets"]] == keys for u in d["by_hospital"])
    assert [b["label"] for b in units[DR]["buckets"]] == ["Formulary", "Out of formulary", "Non-formulary",
                                                          "Consumables", "Not in Item Master"]
    # KR's August, rows 1-6, shares of its 2,050: formulary rows 1 (IP) + 3 (OP) 1,300 / 800 -> 500,
    # 38.46%, 63.41%; row 2 200 / 150 -> 25%, 9.76%; row 4 (blank cost) 100 / 0 -> 100%, 4.88%;
    # consumable row 5 400 / 100 -> 75%, 19.51%; drug not in master row 6 50 / 40 -> 20%, 2.44%
    assert [(b["mrp"], b["cost"], b["margin"], b["margin_pct"], b["share_pct"]) for b in units[KR]["buckets"]] == [
        (1300.0, 800.0, 500.0, 38.46, 63.41), (200.0, 150.0, 50.0, 25.0, 9.76), (100.0, 0.0, 100.0, 100.0, 4.88),
        (400.0, 100.0, 300.0, 75.0, 19.51), (50.0, 40.0, 10.0, 20.0, 2.44)]
    kr = {b["key"]: b for b in units[KR]["buckets"]}
    assert figures(kr["formulary"]) == block(1300, 800, 38.46, 40, 7)
    assert kr["formulary"]["ip"] == block(1000, 600, 40.0, 10, 4)
    assert kr["formulary"]["op"] == block(300, 200, 33.33, 30, 3)
    assert figures(kr["non_formulary"]) == block(100, 0, 100.0, 5, 1, 1, 100)
    # the same cut as the row: KR selected, its own bucket split is the very same
    assert units[KR]["buckets"] == overview(client, Plant=KR, Month=AUG)["by_bucket"]
    # DR: row 7 only, IP formulary 3,000 / 2,000 -> 33.33%, all of DR's revenue; every other bucket 0, 0% of it
    dr = units[DR]["buckets"]
    assert figures(dr[0]) == block(3000, 2000, 33.33, 20, 5) and dr[0]["share_pct"] == 100.0 and dr[0]["op"] == ZERO
    assert all(figures(b) == ZERO and b["share_pct"] == 0.0 for b in dr[1:])
    # Rajkot: row 8 has no MRP, so its margin % and every share are null, never 0 or a divide error
    rj = {b["key"]: b for b in units[RAJKOT]["buckets"]}
    assert figures(rj["non_formulary"]) == block(0, 0, None, 1, 1)
    assert all(b["share_pct"] is None for b in rj.values())
    # every unit's buckets add up to its own row, IP and OP included
    for u in d["by_hospital"]:
        for f in ("mrp", "cost", "margin", "qty", "lines", "cost_missing_lines", "mrp_cost_missing"):
            assert sum(b[f] for b in u["buckets"]) == u[f], (u["plant"], f)
            assert all(sum(b[t][f] for b in u["buckets"]) == u[t][f] for t in ("ip", "op")), (u["plant"], f)
    # the buckets follow the row's filters: GOV in August is KR's rows 1 (formulary 1,000) and 6 (not in
    # master, 50): 95.24% and 4.76% of 1,050; DR has no GOV line in August, so no share
    g = {u["plant"]: u for u in overview(client, Month=AUG, Payer="GOV")["by_hospital"]}
    assert [(b["key"], b["mrp"], b["share_pct"]) for b in g[KR]["buckets"]] == [
        ("formulary", 1000.0, 95.24), ("out_of_formulary", 0.0, 0.0), ("non_formulary", 0.0, 0.0),
        ("consumables", 0.0, 0.0), ("not_in_master", 50.0, 4.76)]
    assert all(b["share_pct"] is None for b in g[DR]["buckets"])
    # OP in August is KR's rows 3 (formulary 300, 75% of 400) and 4 (non-formulary 100, 25%) and Rajkot's
    # row 8: no drug outside the master, so Not in Item Master is left out of every unit
    o = {u["plant"]: u for u in overview(client, Month=AUG, Type="OP")["by_hospital"]}
    assert all([b["key"] for b in u["buckets"]] == keys[:4] for u in o.values())
    assert [(b["mrp"], b["share_pct"]) for b in o[KR]["buckets"]] == [(300.0, 75.0), (0.0, 0.0), (100.0, 25.0),
                                                                     (0.0, 0.0)]
    assert all(b["ip"] == ZERO for b in o[KR]["buckets"])
    # July: KR rows 9 (formulary 800 / 500), 10 (a consumable, 100 / 60) and 11 (non-formulary by July's
    # master, 60 / 30) of 960; no drug outside the master. DR row 12: out of formulary 500 / 450 -> 10%
    j = {u["plant"]: u for u in overview(client, Month=JUL)["by_hospital"]}
    assert [(b["key"], b["mrp"], b["margin_pct"], b["share_pct"]) for b in j[KR]["buckets"]] == [
        ("formulary", 800.0, 37.5, 83.33), ("out_of_formulary", 0.0, None, 0.0),
        ("non_formulary", 60.0, 50.0, 6.25), ("consumables", 100.0, 40.0, 10.42)]
    assert (j[DR]["buckets"][1]["key"], j[DR]["buckets"][1]["margin_pct"], j[DR]["buckets"][1]["share_pct"]) == (
        "out_of_formulary", 10.0, 100.0)
    # a hospital selected: the ranking keeps every unit's buckets as they were
    s = overview(client, Plant=DR, Month=AUG)
    assert [u["buckets"] for u in s["by_hospital"]] == [u["buckets"] for u in d["by_hospital"]]
    assert json.dumps(d, allow_nan=False)


def test_several_months_sum_first_and_have_no_previous_month(client, data):
    a = overview(client, Plant=KR, Month="ALL")
    # rows 1-6 and 9-11: 3010 / 1680 -> 1330 = 44.19% (not the mean of 46.83 and 38.54)
    assert a["headline"] == block(3010, 1680, 44.19, 112, 19, 2, 150)
    assert a["scope"]["months"] == [JUN, JUL, AUG]
    assert a["prev_period"] is None and a["prev"] is None
    assert any("never an average of monthly percentages" in n for n in a["notes"])
    assert overview(client, Plant=KR, Month=f"{AUG},{JUL}")["headline"] == a["headline"]
    # Month defaults to the latest month
    assert overview(client, Plant=KR)["scope"]["months"] == [AUG]


def test_filters_cut_every_view_but_the_ones_that_ignore_them(client, data):
    for params, mrp, cost, pct, buckets in [
        ({"Type": "op"}, 400, 200, 50.0, 4),                       # rows 3, 4
        ({"Segment": "consumables"}, 400, 100, 75.0, 4),           # row 5; no drug outside the master
        ({"Segment": "Onco"}, 1250, 790, 36.8, 5),                 # rows 1, 2, 6 (6 is not in master)
        ({"Segment": "non-onco"}, 400, 200, 50.0, 4),              # rows 3, 4
        ({"Payer": "GOV"}, 1050, 640, 39.05, 5),                   # rows 1, 6
        ({"Payer": "cash(tpa)"}, 100, 0, 100.0, 4),                # row 4
        ({"Basis": "qty"}, 2050, 1090, 46.83, 5),                  # revenue is money: Basis is not a filter
    ]:
        d = overview(client, Plant=KR, Month=AUG, **params)
        h = d["headline"]
        assert (h["mrp"], h["cost"], h["margin_pct"]) == (float(mrp), float(cost), pct), params
        assert len(d["by_bucket"]) == buckets, params
        assert sum(x["mrp"] for x in d["by_bucket"]) == h["mrp"], params
        assert len(d["trend"]) == 3 and d["trend"][-1]["mrp"] == h["mrp"], params
    c = overview(client, Plant=KR, Month=AUG, Segment="consumables")
    assert c["scope"]["segment"] == "consumables"
    assert {x["key"]: x["share_pct"] for x in c["by_bucket"]} == {
        "formulary": 0.0, "out_of_formulary": 0.0, "non_formulary": 0.0, "consumables": 100.0}
    # the unit ranking follows the filters: GOV in August is KR's rows 1 and 6 only
    g = overview(client, Plant=KR, Month=AUG, Payer="GOV")
    assert [(u["plant"], u["mrp"], u["rank"]) for u in g["by_hospital"]] == [(KR, 1050.0, 1), (DR, 0.0, None),
                                                                            (RAJKOT, 0.0, None)]
    # an empty selection: zeros, null ratios, and a note, never an error
    e = overview(client, Plant=KR, Month=AUG, Segment="non-onco", Payer="GOV")
    assert e["headline"] == ZERO and e["prev"] == ZERO
    assert all(x["share_pct"] is None for x in e["by_type"] + e["by_bucket"] + e["by_category"])
    assert any(n.startswith("No pharmacy sales for") for n in e["notes"])
    assert any("this selection has none" in n for n in e["notes"])
    # a unit whose only line has no MRP: margin % and every share are null, not 0 or a divide error
    r = overview(client, Plant=RAJKOT, Month=AUG)
    assert r["headline"] == block(0, 0, None, 1, 1)
    assert [x["share_pct"] for x in r["by_type"]] == [None, None]
    assert not any(n.startswith("No pharmacy sales for") for n in r["notes"])


def test_bad_arguments_and_unknown_or_empty_hospitals(client, data):
    for path in ("/revenue/overview", "/revenue/materials"):
        for params, status in [
            ({"Plant": "NOWHERE"}, 404),
            ({"Month": "Aug-26"}, 400),
            ({"Month": "2026-09"}, 400),
            ({"Month": MAY}, 400),                                  # in the summary only
            ({"Month": ","}, 400),
            ({"Type": "ER"}, 400),
            ({"Segment": "drugs"}, 400),
            ({"Payer": "INSURER"}, 400),
        ]:
            r = client.get(path, params=params)
            assert r.status_code == status, (path, params, r.text)
        # the page serves HIS sales (consumables too), and says so: not "formulary data"
        r = client.get(path, params={"Month": "2025-01"})
        assert r.status_code == 400 and r.json()["detail"].startswith("No sales data for 2025-01."), (path, r.text)
    # the item list is paged at most the Export CSV's 5,000 rows, never "all but the last 5 rows"
    for params in ({"page_size": -5}, {"page_size": 0}, {"page": -1}, {"page_size": 5001}):
        assert client.get("/revenue/materials", params=params).status_code == 400, params
    assert client.get("/revenue/materials", params={"page_size": 5000}).status_code == 200
    # a real hospital code with no sales at all: zeros and a note (not a 404, not every hospital)
    other = next(c for c in sorted(da.load("dim_plant")["plant"].astype(str)) if c not in HIS)
    d = overview(client, Plant=other, Month=AUG)
    assert d["scope"]["plant"] == other and d["headline"] == ZERO
    assert any(n.startswith("No pharmacy sales for") for n in d["notes"])
    assert get(client, "/revenue/materials", Plant=other, Month=AUG) == {"data": [], "total": 0, "sums": {
        "mrp": 0.0, "cost": 0.0, "margin": 0.0, "qty": 0.0, "lines": 0, "cost_missing_lines": 0}}


def test_not_built_is_503(client, data):
    (data / "formulary" / "sales_items.parquet").unlink()
    for path in ("/revenue/overview", "/revenue/materials"):
        r = client.get(path)
        assert r.status_code == 503 and "--from-cache" in r.json()["detail"], (path, r.text)


# ---------- the item list ----------

def test_materials_one_row_per_item_and_bucket(client, data):
    d = get(client, "/revenue/materials", Plant=KR, Month=AUG)
    rows = d["data"]
    # by MRP: 100001 1000, 300001 400, 200001 300, 100002 200, 200003 100, 100009 50
    assert [(r["material"], r["bucket"], r["bucket_label"]) for r in rows] == [
        ("100001", "formulary", "Formulary"), ("300001", "consumables", "Consumables"),
        ("200001", "formulary", "Formulary"), ("100002", "out_of_formulary", "Out of formulary"),
        ("200003", "non_formulary", "Non-formulary"), ("100009", "not_in_master", "Not in Item Master")]
    assert rows[0] == {"material": "100001", "material_desc": "ONCO A 100MG", "generic_name": "Onco generic A",
                       "manufacturer": "Maker 1", "category": "Onco Drugs", "bucket": "formulary",
                       "bucket_label": "Formulary", "mrp": 1000.0, "cost": 600.0, "margin": 400.0,
                       "margin_pct": 40.0, "qty": 10.0, "lines": 4, "hospitals": 1, "cost_missing_lines": 0}
    assert (rows[2]["margin_pct"], rows[4]["margin_pct"], rows[4]["cost_missing_lines"]) == (33.33, 100.0, 1)
    # the sums equal the overview's headline: 2050 / 1090 / 960, qty 88, 15 lines, 2 with no cost
    assert d["total"] == 6
    assert d["sums"] == {"mrp": 2050.0, "cost": 1090.0, "margin": 960.0, "qty": 88.0, "lines": 15,
                         "cost_missing_lines": 2}
    # every hospital: 100001 = KR 1000 / 600 + DR 3000 / 2000, at 2 hospitals; 200003 = KR 100 + Rajkot 0
    every = {(r["material"], r["bucket"]): r for r in get(client, "/revenue/materials", Month=AUG)["data"]}
    assert (every[("100001", "formulary")]["mrp"], every[("100001", "formulary")]["cost"]) == (4000.0, 2600.0)
    assert every[("100001", "formulary")]["hospitals"] == 2 and every[("200003", "non_formulary")]["hospitals"] == 2
    # an item whose class changed between masters has a row per bucket; the name is the newest
    both = get(client, "/revenue/materials", Plant=KR, Month="ALL", page_size=50)["data"]
    para = [(r["bucket"], r["mrp"], r["cost"], r["material_desc"]) for r in both if r["material"] == "200001"]
    assert para == [("formulary", 300.0, 200.0, "PARA 500"), ("non_formulary", 60.0, 30.0, "PARA 500")]
    zero = get(client, "/revenue/materials", Plant=RAJKOT, Month=AUG)["data"]
    assert zero[0]["mrp"] == 0.0 and zero[0]["margin_pct"] is None


def test_materials_facets_scope_sort_and_search(client, data):
    d = get(client, "/revenue/materials", Plant=KR, Month=AUG, Segment="consumables", facets=1)
    assert [f["field"] for f in d["facets"]] == ["category", "bucket_label", "manufacturer"]
    assert {(c["param"], c["value"]) for c in d["scope"]} == {("Plant", KR), ("Month", AUG),
                                                             ("Segment", "consumables")}
    assert [r["material"] for r in d["data"]] == ["300001"]
    f = get(client, "/revenue/materials", Plant=KR, Month=AUG, **{"f.bucket_label": "Formulary"})
    # rows 1 + 3: the sums follow the facet
    assert f["total"] == 2 and f["sums"]["mrp"] == 1300.0 and f["sums"]["margin"] == 500.0
    s = get(client, "/revenue/materials", Plant=KR, Month=AUG, sort_field="margin_pct", sort_order="desc")
    assert [r["material"] for r in s["data"]][:2] == ["200003", "300001"]          # 100%, 75%
    q = get(client, "/revenue/materials", Plant=KR, Month=AUG, global_filter="syringe")
    assert [r["material"] for r in q["data"]] == ["300001"]
    p = get(client, "/revenue/materials", Plant=KR, Month=AUG, page_size=2, page=1)
    assert [r["material"] for r in p["data"]] == ["200001", "100002"] and p["total"] == 6


# ---------- the committed table: Aug-26 network figures ----------

_REAL = Path(settings.FORMULARY_DATA_DIR) / "sales_items.parquet"
_real = pytest.mark.skipif(not _REAL.exists(), reason="sales_items not built — run python -m app.etl.formulary")


@_real
def test_real_aug_26_network_revenue_and_margin(client):
    formulary.clear_caches()
    if AUG not in set(pd.read_parquet(_REAL, columns=["period"])["period"].astype(str)):
        pytest.skip(f"sales_items has no {AUG}")
    d = overview(client, Month=AUG)
    h = d["headline"]
    # HCG's Aug-26 consolidated IP & OP file: total MRP and cost, 875,717 lines, 560 of them with a blank cost
    assert (h["mrp"], h["cost"], h["lines"]) == (943_122_638.17, 556_648_532.40, 875_717)
    assert (h["cost_missing_lines"], h["mrp_cost_missing"]) == (560, 357_062.56)
    assert h["margin"] == 386_474_105.77 and h["margin_pct"] == 40.98
    ip, op = d["by_type"]
    assert (ip["mrp"], ip["cost"]) == (864_743_209.11, 507_635_691.21)
    assert (op["mrp"], op["cost"]) == (78_379_429.06, 49_012_841.19)
    b = {x["key"]: x for x in d["by_bucket"]}
    assert (b["consumables"]["mrp"], b["consumables"]["cost"]) == (211_701_002.96, 92_989_638.66)
    assert "not_in_master" not in b                                    # every drug is in its month's master
    assert sum(x["mrp"] for x in d["by_bucket"]) == pytest.approx(h["mrp"], abs=0.05)
    assert sum(u["mrp"] for u in d["by_hospital"]) == pytest.approx(h["mrp"], abs=0.05)
    kr = next(u for u in d["by_hospital"] if u["plant"] == KR)
    assert (kr["mrp"], kr["cost"]) == (195_657_450.97, 123_988_959.51)
    # KR's August by bucket (no drug is outside its month's master, so four buckets)
    assert [(b["key"], b["mrp"], b["cost"], b["margin_pct"], b["share_pct"]) for b in kr["buckets"]] == [
        ("formulary", 148_772_134.09, 102_061_364.55, 31.4, 76.04),
        ("out_of_formulary", 2_157_384.96, 1_639_445.05, 24.01, 1.1),
        ("non_formulary", 3_322_295.55, 2_364_618.45, 28.83, 1.7),
        ("consumables", 41_405_636.37, 17_923_531.46, 56.71, 21.16)]
    # every unit's buckets add up to its row, and the units' buckets to the network's
    for u in d["by_hospital"]:
        assert [x["key"] for x in u["buckets"]] == list(b), u["plant"]
        for f in ("mrp", "cost", "margin"):
            assert sum(x[f] for x in u["buckets"]) == pytest.approx(u[f], abs=0.05), (u["plant"], f)
            assert sum(x["ip"][f] + x["op"][f] for x in u["buckets"]) == pytest.approx(u[f], abs=0.05)
        assert sum(x["lines"] for x in u["buckets"]) == u["lines"]
        if u["mrp"]:
            assert sum(x["share_pct"] for x in u["buckets"]) == pytest.approx(100, abs=0.05), u["plant"]
    for i, x in enumerate(d["by_bucket"]):
        assert sum(u["buckets"][i]["mrp"] for u in d["by_hospital"]) == pytest.approx(x["mrp"], abs=0.5), x["key"]
    m = get(client, "/revenue/materials", Month=AUG, page_size=1)
    assert m["sums"]["mrp"] == pytest.approx(h["mrp"], abs=0.05) and m["sums"]["lines"] == h["lines"]
