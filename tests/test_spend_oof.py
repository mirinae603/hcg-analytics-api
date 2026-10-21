# tests/test_spend_oof.py — the out-of-formulary margin (Task 3c) on tiny made-up data.
#
# Three layers, every expected number worked out by hand:
#   1. app/etl/spend/oof.py: strength / form keys and the match (same molecule, strength and
#      form, best margin) on hand-made Item Master entries and CASH&TPA prices;
#   2. the spend build on a miniature HCG folder (sales workbook, Item Master with Minor Groups,
#      Payer Master, Main Sheet): which lines count and the published oof_switch rows;
#   3. the /spend/oof/* endpoints and the cash overview's `with_oof` block on published tables.
from __future__ import annotations

import json

import pandas as pd
import pytest
from openpyxl import Workbook

from app.api import spend
from app.core.config import settings
from app.etl.formulary import config as FC
from app.etl.formulary.build import build as build_formulary
from app.etl.formulary.masters import Item
from app.etl.spend import config as C
from app.etl.spend import oof as O
from app.etl.spend.build import build
from app.etl.spend.masters import load_price_list
from tests.test_formulary_pipeline import DASH, HOSPITAL_MAP, OVERRIDES, write_payer_master
from tests.test_spend_api import AUG, JUL, write_tables
from tests.test_spend_pipeline import BIFURCATION, DR, KR, S, write_sales, write_scheme_list


# --- 1. keys and the match -------------------------------------------------------------------

@pytest.mark.parametrize("desc, key", [
    ("ZAVIEDGE 2GM/0.5MG INJ VIAL", "0.5MG|2000MG"),
    ("XAVITAZ 2 GM / 0.5 MG INJ VIAL", "0.5MG|2000MG"),         # spaces and GM vs MG agree
    ("CEFTRIAXONE 1000MG INJ", "1000MG"),
    ("ADRIB 10MG/5ML INJ", "10MG"),                              # a dose fixes the pack
    ("NS 0.9% (DENIS) 250ML PB", "0.9%|250ML"),                  # a % alone does not
    ("INSULIN 100IU/ML 3ML PEN", "100IU|3ML"),                   # nor a dose per mL
    ("TUSQ-DX SYR 100ML BOTTLE", "100ML"),                       # no dose: the volume
    ("EPOFIT 4000 UNITS PFS", "4000IU"),                         # UNITS is a unit
    ("WEPOX 4000 IU PFS", "4000IU"),
    ("HEPARIN 25000 UNITS 5ML VIAL", "25000IU"),
    ("DOLO 650 TAB", "650"),                                     # no unit: the bare number
    ("VIT B12 TAB", ""),                                         # part of a word, not a strength
    ("DOLO TAB 10'S", ""),                                       # a pack count, not a strength
])
def test_strength_key(desc, key):
    assert O.strength_key(desc) == key


@pytest.mark.parametrize("desc, form", [
    ("ZAVIEDGE 2GM INJ VIAL", "INJ"), ("AMPICILLIN 500MG CAP", "CAP"), ("CAPECITABINE 500MG TAB", "TAB"),
    ("ASTHALIN 2.5MG RESP 2.5ML AMP", "NEB"), ("BUDECORT 200MCG ROTACAP", "DPI"),
    ("BUDECORT 200MCG INHALER", "MDI"), ("TUSQ-DX SYR 100ML BOTTLE", "ORAL LIQUID"),
    ("NS 0.9% 250ML PB", "INJ"), ("VALENTAS 50MG", ""),
])
def test_form_key(desc, form):
    assert O.form_key(desc) == form


@pytest.mark.parametrize("desc, kind", [
    ("TRAMADOL 100MG SR TAB", "MR"), ("TRAMADOL 100MG TAB", ""), ("DOXORUBICIN LIPOSOMAL 20MG INJ", "LIPO"),
    ("PACLITAXEL PROTEIN BOUND 100MG INJ", "NAB"),
])
def test_kind_key(desc, kind):
    assert O.kind_key(desc) == kind


def P(code, price, mrp, gst=5.0, cat="P"):
    return code, O.PricedBrand(material=code, material_desc=f"LIST {code}", price=price, mrp=mrp, gst=gst,
                               category=cat)


MASTER = {
    "101001": Item("FORMULARYR", "DRUGA 100MG INJ VIAL", "DRUG A", None),
    "101005": Item("FORMULARYP", "DRUGA PLUS 100 MG INJ", "DRUG A", None),
    "101006": Item("FORMULARYP", "DRUGA 200MG INJ VIAL", "DRUG A", None),     # other strength
    "101008": Item("FORMULARYP", "DRUGA 100MG TAB", "DRUG A", None),          # other form
    "101009": Item("NON FORMULARY", "DRUGA NF 100MG INJ", "DRUG A", None),    # not formulary
    "101010": Item("FORMULARYP", "DRUGA UNPRICED 100MG INJ", "DRUG A", None),  # not in the price list
    "101007": Item("OUT OF FORMULARY", "OOFA 100 MG INJ VIAL", "DRUG A", None),
    "201007": Item("OUT OF FORMULARY", "OOFB 5MG TAB", "DRUG B", None),        # no formulary brand
    "201009": Item("OUT OF FORMULARY", "OOFC 5MG TAB", None, None),            # no molecule
}
# Unit margin = MRP / 1.05 - price: 101001 1200 - 1000 = 200; 101005 1100 - 300 = 800;
# 101006 2000 - 500 = 1500 and 101008 1100 - 100 = 1000 (better, but not the same product).
PRICED = dict([P("101001", 1000.0, 1260.0, cat="R"), P("101005", 300.0, 1155.0), P("101006", 500.0, 2100.0),
               P("101008", 100.0, 1155.0), P("101009", 10.0, 1155.0)])


def test_match_picks_the_best_margin_of_the_same_molecule_strength_and_form():
    m = O.match(MASTER, PRICED).set_index("material")
    assert list(m.index) == ["101007", "201007", "201009"]                # OUT OF FORMULARY only
    a = m.loc["101007"]
    assert (a["matched"], a["candidates"], a["suggested_material"]) == (True, 2, "101005")
    assert (a["suggested_price"], a["suggested_mrp"], a["suggested_gst"], a["suggested_category"]) == \
        (300.0, 1155.0, 5.0, "P")
    assert (a["molecule"], a["strength"], a["form"]) == ("DRUG A", "100MG", "INJ")
    for code in ("201007", "201009"):
        assert not m.loc[code, "matched"] and m.loc[code, "candidates"] == 0
        assert pd.isna(m.loc[code, "suggested_material"])


def test_no_strength_never_matches_and_kinds_stay_apart():
    master = {
        "201001": Item("FORMULARYP", "DOLO TAB 10'S", "PARA", None),
        "201002": Item("OUT OF FORMULARY", "CALPOL TAB 15'S", "PARA", None),          # no strength
        "201003": Item("FORMULARYP", "TRAMA 100MG TAB", "TRAMADOL", None),
        "201004": Item("OUT OF FORMULARY", "TRAMA 100MG SR TAB", "TRAMADOL", None),   # SR vs plain
        "201005": Item("FORMULARYP", "EPO 10000 UNITS PFS", "EPO", None),
        "201006": Item("OUT OF FORMULARY", "EPOFIT 4000 UNITS PFS", "EPO", None),     # other strength
    }
    priced = dict([P("201001", 1.0, 20.0), P("201003", 5.0, 50.0), P("201005", 900.0, 5000.0)])
    m = O.match(master, priced).set_index("material")
    assert not m["matched"].any() and (m["candidates"] == 0).all()


def test_a_formulary_brand_far_dearer_or_cheaper_is_not_a_match():
    # ADRIM-style: plain doxorubicin at MRP ~850 a vial vs a 'same molecule' brand at 12,954
    master = {
        "100001": Item("FORMULARYP", "ADRIB 50MG INJ VIAL", "DOXORUBICIN", None),
        "100002": Item("FORMULARYP", "LIPODOX 50MG INJ VIAL", "DOXORUBICIN", None),
        "100003": Item("OUT OF FORMULARY", "ADRIM 50MG INJ VIAL", "DOXORUBICIN", None),
    }
    priced = dict([P("100001", 600.0, 900.0), P("100002", 10363.2, 12954.0)])
    assert O.match(master, priced).set_index("material").loc["100003", "suggested_material"] == "100002"
    m = O.match(master, priced, unit_mrp={"100003": 848.0}).set_index("material").loc["100003"]
    assert (m["suggested_material"], m["candidates"]) == ("100001", 1)          # LIPODOX is 15x dearer
    m = O.match(master, priced, unit_mrp={"100003": 848.0}, band=20).set_index("material").loc["100003"]
    assert m["candidates"] == 2


def test_equal_margins_pick_the_lowest_code():
    priced = dict([P("101005", 300.0, 1155.0), P("101001", 300.0, 1155.0)])
    assert O.match(MASTER, priced).set_index("material").loc["101007", "suggested_material"] == "101001"


def test_values_by_hand():
    rows = pd.DataFrame([
        {"material": "101007", "qty": 6.0, "actual_cost": 4800.0, "bill_mrp": 7200.0, "matched": True,
         "suggested_price": 300.0, "suggested_mrp": 1155.0, "suggested_gst": 5.0},
        {"material": "201007", "qty": 10.0, "actual_cost": 50.0, "bill_mrp": 75.0, "matched": False,
         "suggested_price": None, "suggested_mrp": None, "suggested_gst": None}])
    v = O.values(rows).set_index("material")
    a = v.loc["101007"]
    # actual = 7200 / 1.05 - 4800 = 2057.142857; proposed = 6 x 800 = 4800
    assert a["actual_margin"] == pytest.approx(7200 / 1.05 - 4800)
    assert a["potential_margin"] == pytest.approx(4800.0)
    assert a["margin_loss"] == pytest.approx(4800 - (7200 / 1.05 - 4800))
    assert (a["used_gst"], a["cash_qty"]) == (5.0, 6.0)
    u = v.loc["201007"]
    assert all(pd.isna(u[c]) for c in ("actual_margin", "potential_margin", "margin_loss", "used_cost_value"))


# --- 2. the build ----------------------------------------------------------------------------

MASTER_ROWS = [  # code, description, Minor Group, flag, category
    (101001, "DRUGA 100MG INJ VIAL", "DRUG A", "FORMULARY", "R"),
    (101005, "DRUGA PLUS 100 MG INJ", "DRUG A", "FORMULARY", "P"),
    (101006, "DRUGA 200MG INJ VIAL", "DRUG A", "FORMULARY", "P"),
    (101007, "OOFA 100 MG INJ VIAL", "DRUG A", "OUT OF FORMULARY", None),
    (201007, "OOFB 5MG TAB", "DRUG B", "OUT OF FORMULARY", None),
    (201002, "PARA 500MG TAB", "PARA", "FORMULARY", "P"),
]
# CASH&TPA rows: used, price, MRP, GST, category, SHARING, suggested, its price, MRP, GST
OOF_CASH = [
    (101001, 1000.0, 1260.0, 5, "R", 0.5, 101005, 300.0, 1155.0, 5),   # a cash switch too
    (101005, 300.0, 1155.0, 5, "P", 0.5, None, None, None, None),
    (101006, 500.0, 2100.0, 5, "P", 1.0, None, None, None, None),
]
SCHEME_ROWS = [(201002, 50.0, "P", 201902, 30.0, False),
               (300001, 7.0, None, 300901, 5.0, False)]            # a SUTURES row, as HCG's list has
OOF_LINES = [
    S(KR, "IP", 101007, 4, 3200.0, None),                                      # cash
    S(KR, "OP", 101007, 2, 1600.0, "Star Health Insurance", doc=("D2", "Dr B")),  # TPA
    S(KR, "IP", 101007, 1, 800.0, "ECHS"),                                      # GOV: not counted
    S(KR, "IP", 201007, 10, 50.0, None),                                        # cash, no match
    S(DR, "IP", 101007, 1, 800.0, None, doc=("D9", "Dr Z")),                    # cash, DR
    S(KR, "IP", 201002, 3, 90.0, "ECHS"),                                       # a scheme switch
]


def write_rich_master(path, rows=MASTER_ROWS):
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Material", "Material Description", "Minor Group", "Manufacturer", "Formulary flagging",
               "Category", "Materialtype"])
    for code, desc, minor, flag, cat in rows:
        ws.append([float(code), desc, minor, "ACME PHARMA", flag, cat, "Onco"])
    wb.save(path)


@pytest.fixture()
def env(tmp_path):
    dash = tmp_path / "raw" / "1829e863" / DASH
    (dash / "IP and OP sales").mkdir(parents=True)
    (dash / "Item Master Sheet").mkdir()
    write_sales(dash / "IP and OP sales" / "IP & OP Consolidated File August 26.xlsx", OOF_LINES)
    write_rich_master(dash / "Item Master Sheet" / "Item master for formulary complience August -26.xlsx")
    write_payer_master(dash / "Copy of Copy of Payer Master Final Sheet.xlsx")
    write_scheme_list(dash / "Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xlsx", rows=SCHEME_ROWS,
                      cash=OOF_CASH)
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / FC.HOSPITAL_MAP_FILE).write_text(HOSPITAL_MAP)
    (ref / FC.PAYER_OVERRIDES_FILE).write_text(OVERRIDES)
    (ref / C.BIFURCATION_FILE).write_text(BIFURCATION)
    dirs = dict(raw=tmp_path / "raw", cache=tmp_path / "cache", reference=ref)
    build_formulary(log=lambda *_: None, data=tmp_path / "formulary", **dirs)
    return dict(dirs=dirs, dash=dash, data=tmp_path / "spend")


def test_the_price_list_keeps_every_priced_brand(env):
    p = load_price_list(env["dash"] / "Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xlsx")
    assert sorted(p) == ["101001", "101005", "101006"]
    assert p["101005"].unit_margin == pytest.approx(800.0)


def test_the_build_publishes_oof_rows_with_hcgs_cash_margins(env):
    rep = build(log=lambda *_: None, data=env["data"], **env["dirs"])
    t = pd.read_parquet(env["data"] / f"{C.OOF_SWITCH}.parquet")
    assert set(t["payer_bucket"]) == {"CASH", "CASH(TPA)"}                     # the GOV line is out
    kr = t[(t["plant"] == FC.KR_PLANT) & t["matched"]]
    assert set(kr["suggested_material"]) == {"101005"} and set(kr["candidates"]) == {2}   # 101001, 101005
    # KR 101007: qty 4 + 2 = 6, bill cost 4800, bill MRP 1.5 x 4800 = 7200 (write_sales)
    assert kr["qty"].sum() == 6.0 and kr["actual_cost"].sum() == 4800.0 and kr["bill_mrp"].sum() == 7200.0
    assert kr["actual_margin"].sum() == pytest.approx(7200 / 1.05 - 4800)
    assert kr["potential_margin"].sum() == pytest.approx(6 * 800.0)
    un = t[~t["matched"]]
    assert list(un["material"]) == ["201007"] and un["actual_margin"].isna().all()
    o = rep["months"][AUG]["kr"]["oof"]
    assert (o["brands"], o["matched_brands"]) == (2, 1)
    assert o["margin_loss"] == pytest.approx(round(4800 - (7200 / 1.05 - 4800), 2))
    assert o["matched_cost_pct"] == pytest.approx(4800 / 4850 * 100, abs=1e-4)
    assert "oof_match" in rep["rules"]
    for c in ("doctor_name", "material_desc"):                                   # names filled in
        assert t[c].notna().all()
    assert not {"UHID", "PATNAME", "SALESINVOICENO"} & {c.upper() for c in t.columns}


def test_a_month_without_an_item_master_publishes_the_full_schema():
    from app.etl.spend.aggregate import _oof_rows
    empty = _oof_rows(None, None)
    assert empty.empty and {"margin_loss", "actual_margin", "potential_margin", "cash_qty"} <= set(empty.columns)


def test_a_month_without_an_item_master_has_no_oof_rows(env):
    rep = build(log=lambda *_: None, data=env["data"], **env["dirs"])
    assert rep["months"][AUG]["item_master"].startswith("Item master")
    from app.etl.spend.build import month_report
    from app.etl.spend.aggregate import MonthResult
    r = month_report(MonthResult(AUG, {}, pd.DataFrame(), {"scheme": {"lines": 1}}), {}, None)
    assert any("no Item Master" in w for w in r["warnings"])


# --- 3. the API --------------------------------------------------------------------------------

# Published oof_switch rows: period, plant, type, payer, doctor, material, matched, qty, bill cost,
# bill MRP; matched brands compare with 101005 (price 300, MRP 1155, GST 5 -> unit margin 800).
OOF_ROWS = [
    (AUG, "HC05", "IP", "CASH", "D1", "101007", True, 4.0, 3200.0, 4800.0),
    (AUG, "HC05", "OP", "CASH(TPA)", "D2", "101007", True, 2.0, 1600.0, 2400.0),
    (AUG, "HC05", "IP", "CASH", "D1", "201007", False, 10.0, 50.0, 75.0),
    (AUG, "HC06", "IP", "CASH", "D9", "101007", True, 1.0, 800.0, 1200.0),
    (JUL, "HC05", "IP", "CASH", "D1", "101007", True, 1.0, 1000.0, 1050.0),
]


def write_oof(folder) -> None:
    rows = []
    for period, plant, typ, payer, doc, mat, matched, q, cost, mrp in OOF_ROWS:
        rows.append({"period": period, "plant": plant, "hospital": {"HC05": "KABHK", "HC06": "KABHD"}[plant],
                     "hospital_name": "X", "patient_type": typ, "category": "Onco Drugs" if mat[0] == "1" else "Other Drugs",
                     "payer_bucket": payer, "doctor_id": doc, "doctor_name": f"Dr {doc}", "material": mat,
                     "material_desc": f"OOF {mat}", "molecule": "DRUG A" if matched else "DRUG B",
                     "strength": "100MG", "form": "INJ", "matched": matched, "candidates": 1 if matched else 0,
                     "suggested_material": "101005" if matched else None,
                     "suggested_desc": "DRUGA PLUS" if matched else None,
                     "suggested_category": "P" if matched else None, "suggested_price": 300.0 if matched else None,
                     "suggested_mrp": 1155.0 if matched else None, "suggested_gst": 5.0 if matched else None,
                     "qty": q, "actual_cost": cost, "bill_mrp": mrp, "lines": 1})
    df = O.values(pd.DataFrame(rows))
    df.to_parquet(folder / "oof_switch.parquet", index=False)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    from tests.test_formulary_api import write_hospital_map
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "spend"))
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary-not-built"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "spend")
    write_oof(tmp_path / "spend")
    write_hospital_map(tmp_path / "reference")
    spend.clear_caches()
    yield tmp_path
    spend.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


# KR Aug matched: qty 6, bill cost 4800, bill MRP 7200 -> actual 7200/1.05 - 4800; proposed 4800.
KR_ACTUAL, KR_PROPOSED = 7200 / 1.05 - 4800, 4800.0


def test_oof_overview_headline_counts_matched_brands_only(client, data):
    d = get(client, "/spend/oof/overview", Plant="HC05", Month=AUG)
    h = d["headline"]
    assert h["actual_margin"] == pytest.approx(KR_ACTUAL, abs=0.01)
    assert h["potential_margin"] == KR_PROPOSED
    assert h["margin_loss"] == pytest.approx(KR_PROPOSED - KR_ACTUAL, abs=0.01)
    assert h["loss_pct"] == pytest.approx((KR_PROPOSED - KR_ACTUAL) / KR_PROPOSED * 100, abs=1e-4)
    assert (h["brands"], h["doctors"], h["qty"]) == (1, 2, 6.0)
    cov = h["coverage"]
    assert (cov["brands"], cov["matched_brands"], cov["unmatched_brands"]) == (2, 1, 1)
    assert (cov["bill_cost"], cov["matched_cost"], cov["unmatched_cost"]) == (4850.0, 4800.0, 50.0)
    assert [u["material"] for u in d["unmatched"]] == ["201007"]
    assert [t["material"] for t in d["top_items"]] == ["101007"]
    assert [r["period"] for r in d["trend"]] == [JUL, AUG]
    assert any("Minor Group" in n for n in d["notes"])


def test_oof_payer_filter_and_hospital_ranking(client, data):
    h = get(client, "/spend/oof/overview", Plant="HC05", Month=AUG, Payer="CASH(TPA)")["headline"]
    assert h["potential_margin"] == 1600.0 and h["qty"] == 2.0                  # 2 x 800
    ranks = get(client, "/spend/oof/overview", Month=AUG)["by_hospital"]
    assert [r["plant"] for r in ranks] == ["HC05", "HC06"]


def test_oof_network_is_the_all_hospitals_figure(client, data):
    whole = get(client, "/spend/oof/overview", Month=AUG)["headline"]["loss_pct"]
    for plant in ("HC05", "HC06"):
        assert get(client, "/spend/oof/overview", Plant=plant, Month=AUG)["network"] == {"loss_pct": whole}


def test_oof_items_list_matched_brands_first_then_unmatched(client, data):
    d = get(client, "/spend/oof/items", Plant="HC05", Month=AUG)
    rows = d["data"]
    assert [r["material"] for r in rows] == ["101007", "201007"]
    a, u = rows
    assert a["matched"] is True and a["suggested_material"] == "101005"
    assert a["suggested_margin"] == pytest.approx(800.0) and a["used_gst"] == 5.0
    assert a["used_margin"] == pytest.approx(KR_ACTUAL / 6, abs=0.01)
    assert a["margin_gap"] == pytest.approx(800.0 - KR_ACTUAL / 6, abs=0.01)
    assert a["share_pct"] == pytest.approx(100.0)
    assert u["matched"] is False and u["margin_loss"] is None and u["actual_cost"] == 50.0
    assert d["sums"]["actual_cost"] == 4850.0
    assert d["sums"]["margin_loss"] == pytest.approx(KR_PROPOSED - KR_ACTUAL, abs=0.01)


def test_oof_doctors(client, data):
    rows = get(client, "/spend/oof/doctors", Plant="HC05", Month=AUG)["data"]
    assert {r["doctor_id"] for r in rows} == {"D1", "D2"}
    one = get(client, "/spend/oof/doctors", Plant="HC05", Month=AUG, Material="101007")["data"]
    assert sum(r["potential_margin"] for r in one) == KR_PROPOSED


def test_the_cash_overview_adds_cash_and_oof_like_hcgs_table(client, data):
    d = get(client, "/spend/cash/overview", Plant="HC05", Month=AUG)
    w = d["with_oof"]
    cash = d["headline"]
    assert w["cash"]["margin_loss"] == cash["margin_loss"]
    assert w["oof"]["margin_loss"] == pytest.approx(KR_PROPOSED - KR_ACTUAL, abs=0.01)
    assert w["total"]["actual_margin"] == pytest.approx(cash["actual_margin"] + KR_ACTUAL, abs=0.02)
    assert w["total"]["potential_margin"] == pytest.approx(cash["potential_margin"] + KR_PROPOSED, abs=0.02)
    tot = w["total"]
    assert tot["loss_pct"] == pytest.approx(tot["margin_loss"] / tot["potential_margin"] * 100, abs=1e-3)
    aug = [r for r in w["trend"] if r["period"] == AUG][0]
    assert aug["loss"] == pytest.approx(aug["cash_loss"] + aug["oof_loss"], abs=0.02)
    assert w["coverage"]["unmatched_brands"] == 1


def test_an_empty_selection_gives_zeros_not_an_error(client, data):
    d = get(client, "/spend/oof/overview", Plant="HC06", Month=AUG, Type="OP")
    assert d["headline"]["margin_loss"] == 0.0 and d["headline"]["loss_pct"] is None
    assert d["top_items"] == [] and d["unmatched"] == []
    assert any("No cash or TPA patient" in n for n in d["notes"])
    assert get(client, "/spend/oof/items", Plant="HC06", Month=AUG, Type="OP")["data"] == []


def test_a_brand_matched_in_one_month_shows_that_match(client, data):
    t = pd.read_parquet(data / "spend" / "oof_switch.parquet")
    jul = t[(t["period"] == JUL) & (t["material"] == "101007")].copy()
    for c in ("suggested_material", "suggested_desc", "suggested_category"):
        jul[c] = None
    for c in ("suggested_price", "suggested_mrp", "suggested_gst"):
        jul[c] = float("nan")
    jul["matched"], jul["candidates"] = False, 0
    jul["period"] = "2026-09"                     # a later month in which the match was lost
    O.values(pd.concat([t, jul], ignore_index=True)).to_parquet(data / "spend" / "oof_switch.parquet", index=False)
    spend.clear_caches()
    rows = get(client, "/spend/oof/items", Plant="HC05", Month=f"{JUL},{AUG}")["data"]
    a = [r for r in rows if r["material"] == "101007"][0]
    assert a["suggested_material"] == "101005" and a["suggested_margin"] == pytest.approx(800.0)
    cov = get(client, "/spend/oof/overview", Plant="HC05", Month=f"{JUL},{AUG}")["headline"]["coverage"]
    assert cov["matched_brands"] + cov["unmatched_brands"] == cov["brands"]


def test_an_oof_table_without_margins_counts_as_not_built(client, data):
    t = pd.read_parquet(data / "spend" / "oof_switch.parquet")
    t.drop(columns=["actual_margin", "potential_margin", "margin_loss"]).to_parquet(
        data / "spend" / "oof_switch.parquet", index=False)
    spend.clear_caches()
    assert get(client, "/spend/cash/overview", Plant="HC05", Month=AUG)["with_oof"] is None
    assert client.get("/spend/oof/overview", params={"Plant": "HC05", "Month": AUG}).status_code == 503


def test_without_the_oof_table_cash_still_works_and_oof_says_rebuild(client, data):
    (data / "spend" / "oof_switch.parquet").unlink()
    spend.clear_caches()
    assert get(client, "/spend/cash/overview", Plant="HC05", Month=AUG)["with_oof"] is None
    r = client.get("/spend/oof/overview", params={"Plant": "HC05", "Month": AUG})
    assert r.status_code == 503 and "out-of-formulary" in r.json()["detail"]
    assert get(client, "/spend/meta")["oof_built"] is False
