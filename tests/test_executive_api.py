# tests/test_executive_api.py — /executive/overview: HCG's pharmacy on one screen, every figure
# taken from the section that owns it.
#
# The fixture is the sections' own test fixtures side by side, written by their own writers:
# 13 sales_items rows and the formulary summary built from them (tests/test_revenue_api.py:
# Jun / Jul / Aug, plus a May the summary holds and sales do not), the spend tables
# (tests/test_spend_api.py: Jul / Aug), the out-of-formulary purchases (tests/
# test_spend_oof_purchase_api.py: Jul / Aug) and the price-check purchases (tests/
# test_purchase_insights_api.py CHECK_ROWS: Jul / Aug). Only HCG's unit list and budgets are
# written here. Every expected number was worked out by hand from those rows (comments show the
# arithmetic), not by re-running the API. The last tests read the COMMITTED tables and set the
# overview beside each section's own endpoint, skipped when a table has not been built.
from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from app.api import executive as E
from app.api import formulary
from app.api import purchase_insights as PI
from app.api import revenue
from app.api import spend
from app.core import data_access as da
from app.core.config import settings
from app.etl.purchase import config as PC
from app.etl.purchase import oof as PO
from app.etl.purchase import prices as PP

from tests.test_formulary_api import write_hospital_map
from tests.test_purchase_insights_api import CHECK_BRANDS, CHECK_ROWS, CHECK_VENDORS, write_prices
from tests.test_revenue_api import ROWS as SALES_ROWS
from tests.test_revenue_api import SUMMARY_KEY, sales_items_frame
from tests.test_revenue_api import write_tables as write_sales
from tests.test_spend_api import write_tables as write_spend
from tests.test_spend_oof_purchase_api import ROWS as OOF_ROWS
from tests.test_spend_oof_purchase_api import _row as oof_row
from tests.test_spend_oof_purchase_api import write_oof_tables

MAY, JUN, JUL, AUG = "2026-05", "2026-06", "2026-07", "2026-08"
KR, DR, RAJKOT, GOLKULA = "HC05", "HC06", "HS01", "HC03"
KPI_KEYS = ["sales", "margin_pct", "compliance", "oof_share", "losses", "extra_paid"]


def write_mis(folder: Path) -> None:
    """HCG's unit list (KR, DR and Rajkot in its PAN order, MSR-Golkula outside it, Rajkot with
    no region, as HCG leaves Indore's blank) and FY 26-27 budgets: KR 100%, DR 98%, Rajkot 85%."""
    pd.DataFrame([
        {"plant": KR, "unit_name": "KR BANGALORE", "region": "Karnataka", "pan_order": 2, "in_pan": True},
        {"plant": DR, "unit_name": "DR-BANGALORE", "region": "Karnataka", "pan_order": 3, "in_pan": True},
        {"plant": RAJKOT, "unit_name": "RAJKOT", "region": None, "pan_order": 20, "in_pan": True},
        {"plant": GOLKULA, "unit_name": "MSR-GOLKULA", "region": "Karnataka", "pan_order": None, "in_pan": False},
    ]).astype({"pan_order": "Int64"}).to_parquet(folder / "hcg_mis_units.parquet", index=False)
    pd.DataFrame([{"plant": p, "period": "2026-04", "budget_pct": b, "hcg_actual_pct": b, "hcg_formula": "pooled",
                   "pooled_pct": b} for p, b in ((KR, 100.0), (DR, 98.0), (RAJKOT, 85.0))]
                 ).to_parquet(folder / "hcg_mis_compliance.parquet", index=False)


def _clear() -> None:
    formulary.clear_caches()                # revenue's and the executive's results are cached there too
    spend.clear_caches()
    PI.clear_caches()


@pytest.fixture()
def data(tmp_path, monkeypatch):
    for key, sub in (("FORMULARY_DATA_DIR", "formulary"), ("SPEND_DATA_DIR", "spend"),
                     ("PURCHASE_DATA_DIR", "purchase"), ("REFERENCE_DIR", "reference")):
        monkeypatch.setattr(settings, key, str(tmp_path / sub))
    write_sales(tmp_path / "formulary")
    write_mis(tmp_path / "formulary")
    write_spend(tmp_path / "spend")
    write_oof_tables(tmp_path / "purchase")
    write_prices(tmp_path / "purchase", CHECK_ROWS, CHECK_BRANDS, CHECK_VENDORS)
    write_hospital_map(tmp_path / "reference")
    _clear()
    yield tmp_path
    _clear()


def _finite_everywhere(x) -> bool:
    if isinstance(x, float):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(_finite_everywhere(v) for v in x.values())
    if isinstance(x, list):
        return all(_finite_everywhere(v) for v in x)
    return True


def ov(client, **params) -> dict:
    r = client.get("/executive/overview", params=params)
    assert r.status_code == 200, r.text
    d = r.json()
    assert _finite_everywhere(d)
    return d


def kpis(d: dict) -> dict:
    return {k["key"]: k for k in d["kpis"]}


def figures(k: dict) -> tuple:
    return k["value"], k["prev"], k["delta"]


def spark(k: dict) -> list:
    return [(s["label"], s["value"]) for s in k["spark"]]


# ---------- the default month: every KPI beside the previous month ----------

def test_default_month_kpis_by_hand(client, data):
    d = ov(client)
    assert list(d) == ["scope", "months", "kpis", "trend", "bridge", "ip_op", "units", "mix", "treemap", "flow",
                       "top", "insights", "actions", "notes", "data_as_of"]
    assert d["scope"] == {"plant": "ALL", "plant_name": "All Hospitals", "his_name": None, "month": AUG,
                          "month_label": "Aug-26", "prev_month": JUL, "prev_label": "Jul-26", "type": "ALL",
                          "segment": "ALL", "basis": "value", "budget_judged": True}
    # sales months Jun-Aug and out-of-formulary purchase months Jul-Aug; the summary's May is no page month
    assert d["months"] == [{"period": JUN, "label": "Jun-26", "sales": True, "purchases": False},
                           {"period": JUL, "label": "Jul-26", "sales": True, "purchases": True},
                           {"period": AUG, "label": "Aug-26", "sales": True, "purchases": True}]
    assert [k["key"] for k in d["kpis"]] == KPI_KEYS
    k = kpis(d)
    # sales rows 1-8: MRP 5,050 against July's rows 9-12, 1,460: +245.89%; June row 13, 700
    assert figures(k["sales"]) == (5050.0, 1460.0, 245.89)
    assert (k["sales"]["unit"], k["sales"]["delta_unit"], k["sales"]["good_when"], k["sales"]["source"]) == \
        ("inr", "pct", "up", "sales")
    assert spark(k["sales"]) == [("Jun-26", 700.0), ("Jul-26", 1460.0), ("Aug-26", 5050.0)]
    assert k["sales"]["label"] == "Revenue (MRP)"                    # the Revenue page's name for it
    assert k["sales"]["note"] == ("Total MRP of the HIS pharmacy sales lines, IP and OP: HCG's revenue. The HIS "
                                  "file does not say whether its MRP and cost include GST.")
    # margin: (5,050 - 3,090) / 5,050 = 38.81%; July (1,460 - 1,040) / 1,460 = 28.77%; June 300 / 700
    assert figures(k["margin_pct"]) == (38.81, 28.77, 10.04)
    assert spark(k["margin_pct"]) == [("Jun-26", 42.86), ("Jul-26", 28.77), ("Aug-26", 38.81)]
    # D6: rows 4 and 5 have a blank cost price (1 line each, MRP 100 + 50): counted at cost 0, and said
    assert k["margin_pct"]["tag"] == "incl. 2 lines with no cost"
    assert k["margin_pct"]["note"].endswith("not profit after tax. The HIS file does not say whether its MRP and "
                                            "cost include GST. 2 lines with a blank cost price (MRP ₹ "
                                            "150) count as cost 0, as HCG's sheet does, so that MRP is margin in full.")
    assert revenue.NOTE_GST in d["notes"] and ("2 lines in this selection have a blank cost price in the HIS file, "
                                               "with MRP ₹150.00.") in " ".join(d["notes"])
    # drug cost, August: A = 600 + 200 + 2,000, B = 150, C = 0 -> 2,800 / 2,950 = 94.92%;
    # July A 500, B 450, C 30 -> 52.63%; every formulary month, May (the summary's copy of June) too
    assert figures(k["compliance"]) == (94.92, 52.63, 42.29)
    assert (k["compliance"]["delta_unit"], k["compliance"]["target"], k["compliance"]["status"]) == ("pts", None, None)
    assert spark(k["compliance"]) == [("May-26", 100.0), ("Jun-26", 100.0), ("Jul-26", 52.63), ("Aug-26", 94.92)]
    assert "budgets per unit" in k["compliance"]["note"]
    # out of formulary's share of A + B + C: 150 / 2,950 = 5.08%; July 450 / 980 = 45.92%
    assert figures(k["oof_share"]) == (5.08, 45.92, -40.84) and k["oof_share"]["good_when"] == "down"
    # cash 1,725 + 1,035 + 420 + 680 = 3,860; scheme 1,200 + 600 + 200 + 8 + 50 + 600 = 2,658;
    # out of formulary KR 2,820 + DR (600 - 2,200) = 1,220 -> 7,738. July 280 + 1,880 + 1,800 = 3,960
    assert figures(k["losses"]) == (7738.0, 3960.0, 95.4)
    assert spark(k["losses"]) == [("Jul-26", 3960.0), ("Aug-26", 7738.0)]
    assert k["losses"]["note"] == ("Cash & TPA ₹ 3,860 + scheme ₹ 2,658 + out of formulary ₹ 1,220. Out of "
                                   "formulary is counted on purchases.")
    # two parts on the sales lines and one on purchases: the KPI names both places
    assert (k["losses"]["source"], k["losses"]["sources"]) == ("sales", ["sales", "purchases"])
    assert [k[x]["sources"] for x in KPI_KEYS if x != "losses"] == [["sales"]] * 4 + [["purchases"]]
    assert [k[x]["tag"] for x in KPI_KEYS if x != "margin_pct"] == [None] * 5
    # trastuzumab above August's usual cheapest 10,000: 1,08,000 - 90,000 + 3,00,000 - 2,00,000;
    # July against its own 8,000: 1,40,000 - 80,000
    assert figures(k["extra_paid"]) == (118000.0, 60000.0, 96.67)
    assert (k["extra_paid"]["source"], k["extra_paid"]["good_when"]) == ("purchases", "down")
    assert spark(k["extra_paid"]) == [("Jul-26", 60000.0), ("Aug-26", 118000.0)]
    assert k["extra_paid"]["label"] == "Paid above the usual cheapest"
    # the usual cheapest in the Prices page's words, at the prices answer's own share (prices.USUAL_SHARE)
    assert k["extra_paid"]["note"] == (
        f"Paid above each product's usual cheapest price (the price reached once at least "
        f"{PP.USUAL_SHARE * 100:g}% of its units have been counted, cheapest purchase first, so one odd cheap "
        f"purchase cannot set it), on purchases.")
    # a link's page agrees for the same month and filters, not whatever it shows by default
    assert "a figure matches that page for the same month and filters" in d["notes"][0]
    assert "Prices holds every purchase month" in d["notes"][0]
    assert all(k[x]["target"] is None and k[x]["status"] is None for x in KPI_KEYS)
    assert "Changes are against Jul-26, the previous month with sales." in d["notes"]
    # every section holds August: the purchases are of the same month as the sales
    assert ("Out of formulary and extra paid are counted on purchases (goods receipts) of the same months, not on "
            "sales.") in d["notes"]
    assert E.NOTE_TOP["value"] in d["notes"] and E.NOTE_BOTTOM in d["notes"]
    assert d["data_as_of"] == "2026-10-30T12:00:00+00:00"
    assert ov(client, Month=AUG) == d


def test_bridge_ip_op_mix_and_trend(client, data):
    d = ov(client, Month=AUG)
    # margin earned 1,960, then each part of what was missed, then the margin with every switch
    assert [(s["key"], s["value"], s["kind"], s["source"]) for s in d["bridge"]["steps"]] == [
        ("earned", 1960.0, "total", "sales"), ("cash", 3860.0, "delta", "sales"), ("scheme", 2658.0, "delta", "sales"),
        ("oof", 1220.0, "delta", "purchases"), ("possible", 9698.0, "total", "sales")]
    assert "indication" in d["bridge"]["note"] and "Left out" not in d["bridge"]["note"]
    # IP rows 1, 2, 5, 6, 7: 4,650 / 2,890 -> 37.85%, 92.08% of MRP; OP rows 3, 4, 8: 400 / 200
    assert d["ip_op"] == [
        {"type": "IP", "label": "Inpatient (IP)", "mrp": 4650.0, "margin": 1760.0, "margin_pct": 37.85, "qty": 73.0,
         "share_pct": 92.08},
        {"type": "OP", "label": "Outpatient (OP)", "mrp": 400.0, "margin": 200.0, "margin_pct": 50.0, "qty": 36.0,
         "share_pct": 7.92}]
    assert d["mix"] == {"formulary": 2800.0, "out_of_formulary": 150.0, "non_formulary": 0.0, "total": 2950.0,
                        "formulary_pct": 94.92, "oof_pct": 5.08, "nf_pct": 0.0}
    assert d["trend"] == [
        {"period": JUN, "label": "Jun-26", "ip_mrp": 700.0, "op_mrp": 0.0, "mrp": 700.0, "cost": 400.0,
         "margin": 300.0, "margin_pct": 42.86, "compliance_pct": 100.0, "qty": 7.0},
        {"period": JUL, "label": "Jul-26", "ip_mrp": 1300.0, "op_mrp": 160.0, "mrp": 1460.0, "cost": 1040.0,
         "margin": 420.0, "margin_pct": 28.77, "compliance_pct": 52.63, "qty": 29.0},
        {"period": AUG, "label": "Aug-26", "ip_mrp": 4650.0, "op_mrp": 400.0, "mrp": 5050.0, "cost": 3090.0,
         "margin": 1960.0, "margin_pct": 38.81, "compliance_pct": 94.92, "qty": 109.0}]


def test_units_in_hcgs_order_with_every_column(client, data):
    units = ov(client, Month=AUG)["units"]
    # HCG's PAN order (KR 2, DR 3, Rajkot 20). MSR-Golkula is in HCG's history sheets only (not
    # in the PAN list) and has nothing in August: it is left out, as the hospital selector leaves
    # it out (Plant=HC03 is an unknown hospital)
    assert [u["plant"] for u in units] == [KR, DR, RAJKOT]
    kr, dr, rajkot = units
    # KR: A 800, B 150, C 0 -> 84.21%, 15.79 pts under its July 100% (A 500, no B); 15.79 pts short of
    # its 100% budget, beyond the 2-pt band: below. Its own parts of each loss and of extra paid.
    assert kr == {"plant": KR, "name": "HCG KR, Bangalore", "his_name": "HCG KR, Bangalore", "region": "Karnataka",
                  "pan_order": 2, "mrp": 2050.0, "margin_pct": 46.83, "compliance_pct": 84.21, "budget_pct": 100.0,
                  "budget_note": None, "status": "below", "delta_pts": -15.79, "formulary_share_pct": 84.21,
                  "oof_share_pct": 15.79, "nf_share_pct": 0.0, "oof_loss": 2820.0, "cash_loss": 3180.0,
                  "scheme_missed": 2058.0, "extra_paid": 118000.0, "selected": False}
    # DR: all formulary in August (100%), all out of formulary in July (0%); meets 98%; its brands
    # bought out of formulary earned 1,600 more than the suggestions (net, as HCG adds it)
    assert (dr["compliance_pct"], dr["budget_pct"], dr["status"], dr["delta_pts"]) == (100.0, 98.0, "met", 100.0)
    assert (dr["oof_loss"], dr["cash_loss"], dr["scheme_missed"], dr["extra_paid"]) == (-1600.0, 680.0, 600.0, 0.0)
    # Rajkot: one free line, no MRP and no A + B: no %, no share, no status; a unit no section lists
    # has nothing missed there (0), not a missing figure
    assert (rajkot["mrp"], rajkot["margin_pct"], rajkot["compliance_pct"], rajkot["status"]) == (0.0, None, None, None)
    assert (rajkot["region"], rajkot["budget_pct"], rajkot["formulary_share_pct"]) == (None, 85.0, None)
    assert (rajkot["oof_loss"], rajkot["cash_loss"], rajkot["scheme_missed"], rajkot["extra_paid"]) == (0.0,) * 4
    # the table ignores the hospital selector and marks the one chosen
    assert [u["selected"] for u in ov(client, Month=AUG, Plant="KABHK")["units"]] == [True, False, False]


def _mis_in_pan(folder: Path, **in_pan) -> None:
    """HCG's unit list as write_mis writes it, with `in_pan` changed for the plants named."""
    p = folder / "hcg_mis_units.parquet"
    units = pd.read_parquet(p)
    for plant, flag in in_pan.items():
        units.loc[units["plant"] == plant, "in_pan"] = flag
    units.to_parquet(p, index=False)
    _clear()


def test_a_unit_outside_the_pan_list_is_listed_only_with_figures(client, data):
    # in HCG's PAN list, MSR-Golkula is listed with nothing to show: HCG's own name, no order, no
    # budget, no sales
    _mis_in_pan(data / "formulary", HC03=True)
    golkula = ov(client, Month=AUG)["units"][-1]
    assert (golkula["plant"], golkula["name"], golkula["his_name"], golkula["pan_order"], golkula["mrp"],
            golkula["budget_pct"]) == (GOLKULA, "MSR-GOLKULA", None, None, 0.0, None)
    # Rajkot taken out of the PAN list still has an August sales line (row 8): it stays
    _mis_in_pan(data / "formulary", HC03=False, HS01=False)
    assert [u["plant"] for u in ov(client, Month=AUG)["units"]] == [KR, DR, RAJKOT]
    # in June Rajkot has no sales, no drug consumption, no purchase and nothing missed: left out
    assert [u["plant"] for u in ov(client, Month=JUN)["units"]] == [KR, DR]


def test_treemap_flow_and_top_lists(client, data):
    d = ov(client, Month=AUG)
    # MRP by material type, then HCG's bucket: onco rows 1, 7 (formulary), 2 (out), 6 (not in master)
    assert d["treemap"] == [
        {"key": "onco", "label": "Onco", "value": 4250.0, "children": [
            {"key": "formulary", "label": "Formulary", "value": 4000.0},
            {"key": "out_of_formulary", "label": "Out of formulary", "value": 200.0},
            {"key": "not_in_master", "label": "Not in Item Master", "value": 50.0}]},
        {"key": "non-onco", "label": "Non-Onco", "value": 400.0, "children": [
            {"key": "formulary", "label": "Formulary", "value": 300.0},
            {"key": "non_formulary", "label": "Non-formulary", "value": 100.0}]},
        {"key": "consumables", "label": "Consumables", "value": 400.0, "children": [
            {"key": "consumables", "label": "Consumables", "value": 400.0}]}]
    f = d["flow"]
    # DR 3,000 and KR 2,050 (Rajkot's MRP is 0, so no node and no "Other units")
    assert [(n["id"], n["column"]) for n in f["nodes"]] == [
        ("unit:HC06", 0), ("unit:HC05", 0), ("type:onco", 1), ("type:non-onco", 1), ("type:consumables", 1),
        ("bucket:formulary", 2), ("bucket:out_of_formulary", 2), ("bucket:non_formulary", 2),
        ("bucket:consumables", 2), ("bucket:not_in_master", 2)]
    assert [(x["source"], x["target"], x["value"]) for x in f["links"]] == [
        ("unit:HC06", "type:onco", 3000.0), ("unit:HC05", "type:onco", 1250.0),
        ("unit:HC05", "type:non-onco", 400.0), ("unit:HC05", "type:consumables", 400.0),
        ("type:onco", "bucket:formulary", 4000.0), ("type:onco", "bucket:out_of_formulary", 200.0),
        ("type:onco", "bucket:not_in_master", 50.0), ("type:non-onco", "bucket:formulary", 300.0),
        ("type:non-onco", "bucket:non_formulary", 100.0), ("type:consumables", "bucket:consumables", 400.0)]
    # consumption ranked on cost, as the formulary pages rank it: of 3,090
    assert [(i["material"], i["value"], i["qty"], i["share_pct"]) for i in d["top"]["items"]] == [
        ("100001", 2600.0, 30.0, 84.14), ("200001", 200.0, 30.0, 6.47), ("100002", 150.0, 2.0, 4.85),
        ("300001", 100.0, 40.0, 3.24), ("100009", 40.0, 1.0, 1.29)]
    assert d["top"]["items"][1]["material_desc"] == "PARA 500" and d["top"]["items"][0]["generic_name"] == \
        "Onco generic A"
    assert [(u["plant"], u["value"], u["share_pct"]) for u in d["top"]["units"]] == [(DR, 2000.0, 64.72),
                                                                                     (KR, 1090.0, 35.28)]
    # the other end: the lowest margin % of 5,050 at MRP (each item has at least 0.1% of it, 5.05):
    # 100009 (50 - 40) / 50 = 20%; 100002 50 / 200 = 25%; 200001 100 / 300; 100001 (1,000 + 3,000 -
    # 600 - 2,000) / 4,000 = 35%; the syringe 300 / 400 = 75%; 200003 at a blank cost (row 4) and free
    # (row 8), 100%
    bottom = d["top"]["bottom"]
    assert [(i["material"], i["mrp"], i["margin"], i["margin_pct"], i["share_pct"]) for i in bottom["items"]] == [
        ("100009", 50.0, 10.0, 20.0, 0.99), ("100002", 200.0, 50.0, 25.0, 3.96), ("200001", 300.0, 100.0, 33.33, 5.94),
        ("100001", 4000.0, 1400.0, 35.0, 79.21), ("300001", 400.0, 300.0, 75.0, 7.92),
        ("200003", 100.0, 100.0, 100.0, 1.98)]
    assert bottom["items"][0]["material_desc"] == "ONCO NEW INJ"
    # units: DR 1,000 / 3,000 = 33.33%, KR 960 / 2,050 = 46.83%; Rajkot sold nothing at MRP
    assert [(u["plant"], u["margin_pct"], u["share_pct"]) for u in bottom["units"]] == [(DR, 33.33, 59.41),
                                                                                        (KR, 46.83, 40.59)]
    # one hospital: its items and flow; the unit ranking still every hospital
    k = ov(client, Month=AUG, Plant=KR)
    assert [n["id"] for n in k["flow"]["nodes"]][:2] == ["unit:HC05", "type:onco"]
    assert [i["material"] for i in k["top"]["items"]][0] == "100001" and len(k["top"]["units"]) == 2
    assert k["treemap"][0]["value"] == 1250.0                                    # KR's onco 1,000 + 200 + 50
    # KR's own items, each over 0.1% of its 2,050: 100001 is (1,000 - 600) / 1,000 = 40% there
    assert [i["material"] for i in k["top"]["bottom"]["items"]] == ["100009", "100002", "200001", "100001", "300001",
                                                                    "200003"]
    assert [u["plant"] for u in k["top"]["bottom"]["units"]] == [DR, KR]


def test_the_bottom_lists_leave_out_a_line_under_the_share(monkeypatch, client, data):
    # at 2%, 100009 (50 of 5,050 = 0.99%) and 200003 (1.98%) are left out
    monkeypatch.setattr(E, "BOTTOM_MIN_SHARE_PCT", 2.0)
    _clear()
    d = ov(client, Month=AUG)
    assert [i["material"] for i in d["top"]["bottom"]["items"]] == ["100002", "200001", "100001", "300001"]
    # nothing sold: no bottom list, never a divide error
    assert ov(client, Month=JUL, Segment="consumables", Type="IP")["top"]["bottom"] == {"items": [], "units": []}


# ---------- insights and actions ----------

def test_insights_and_actions_by_hand(client, data):
    d = ov(client, Month=AUG)
    ins = {i["id"]: i for i in d["insights"]}
    # seven rules apply; the six kept are the alerts first, then the better news, in KPI order
    assert [i["id"] for i in d["insights"]] == ["units_below_budget", "cash_missed", "prices_extra", "sales_margin",
                                                "compliance", "oof_missed"]
    assert [i["tone"] for i in d["insights"]] == ["alert", "alert", "alert", "up", "up", "up"]
    assert ins["units_below_budget"]["text"] == (
        "1 of 2 units with a budget missed it in Aug-26 (1 more than 2 pts short as HCG's MIS prints it). Furthest: "
        "HCG KR, Bangalore at 84.21% against 100%, 15.79 pts short.")
    assert ins["cash_missed"]["text"] == (
        "Cash & TPA patients: ₹ 3,860 of margin missed in Aug-26 (65.65% of the margin possible). The biggest "
        "single switch: ONCO X 100MG → ONCO X PRIME, ₹ 3,440.")
    assert ins["prices_extra"]["text"] == (
        "₹ 1.18 L was paid above the usual cheapest price in Aug-26 (22.87% of the purchases compared), most of it "
        "on TRASTUZUMAB 440MG INJ (₹ 1.18 L). 2 purchases were under half the usual cheapest: worth a check.")
    assert ins["sales_margin"]["text"] == (
        "Revenue was ₹ 5,050 at MRP in Aug-26, up 245.89% from Jul-26, with a margin of 38.81% (28.77% in Jul-26). "
        "IP earned 37.85% on ₹ 4,650 and OP 50.00% on ₹ 400.")
    assert ins["compliance"]["text"] == ("Formulary compliance across all hospitals was 94.92% in Aug-26. That is up "
                                         "42.29 pts from Jul-26 (52.63%).")
    # net 1,220 against July's 1,800; KR is every rupee missed at a unit with a loss (DR's is negative)
    assert ins["oof_missed"]["text"] == (
        "Out-of-formulary purchases missed ₹ 1,220 of margin in Aug-26, down ₹ 580 from Jul-26 (32.22%). HCG KR, "
        "Bangalore missed the most: ₹ 2,820 (100.00% of the margin missed at units with a loss).")
    # each link opens its page on the same month
    assert [i["link"] for i in d["insights"]] == [
        "/formulary?view=overview&month=2026-08", "/spend?section=cash&view=overview&month=2026-08",
        "/spend?section=prices&month=2026-08", "/revenue?month=2026-08", "/formulary?view=overview&month=2026-08",
        "/spend?section=oof&view=overview&month=2026-08"]
    # biggest rupees first, then the follow-up (no rupees of its own); the check of the low prices
    # is the fifth and is cut
    assert [(a["id"], a["impact"]) for a in d["actions"]] == [
        ("negotiate_price", 118000.0), ("oof_switch", 3600.0), ("cash_switch", 3440.0), ("follow_up_units", None)]
    assert [a["text"] for a in d["actions"]] == [
        "Negotiate the price of TRASTUZUMAB 440MG INJ: ₹ 1.18 L was paid above its usual cheapest of ₹10,000 a unit "
        "in Aug-26.",
        "Buy ONCO FORM A 100MG INJ in place of ONCO OOF A 100MG INJ: ₹ 3,600 of margin missed on out-of-formulary "
        "purchases in Aug-26.",
        "Prescribe ONCO X PRIME in place of ONCO X 100MG for cash & TPA patients: ₹ 3,440 of margin missed in Aug-26.",
        "Follow up on formulary compliance with HCG KR, Bangalore (15.79 pts short of its 100% budget)."]
    assert list(d["actions"][0]) == ["id", "text", "impact", "link"]


def test_the_units_view_keeps_the_info_insight_and_the_check(client, data):
    d = ov(client, Month=AUG, Basis="qty")
    # no budget is judged on units, so the units-below-budget rule leaves itself out and the
    # seventh rule fits: DR's brands bought out of formulary earned more than the suggestions
    assert [i["id"] for i in d["insights"]] == ["cash_missed", "prices_extra", "sales_margin", "compliance",
                                                "oof_missed", "oof_better_units"]
    better = d["insights"][-1]
    assert better["tone"] == "info" and better["text"] == (
        "At 1 unit (HCG DR, Bangalore) the out-of-formulary brands bought earned ₹ 1,600 more margin than the "
        "suggested ones in Aug-26.")
    assert d["insights"][2]["text"].startswith("109 units were sold in Aug-26, up 275.86% from Jul-26, with a "
                                               "margin of 38.81%")
    assert d["insights"][3]["text"] == ("Formulary compliance on units sold across all hospitals was 96.77% in Aug-26. "
                                        "That is up 35.23 pts from Jul-26 (61.54%).")
    # no follow-up without a budget; the check of the two low groups now fits
    assert [a["id"] for a in d["actions"]] == ["negotiate_price", "oof_switch", "cash_switch", "check_low_prices"]
    assert d["actions"][-1]["text"] == ("Check the 2 purchases bought under half the usual cheapest (₹ 8,000 in "
                                        "Aug-26): a keying slip or a free or bonus supply is likely.")
    # it opens the Prices page on its list of purchases to check, for the same month
    assert d["actions"][-1]["link"] == "/spend?section=prices&month=2026-08&check=1"
    # the units view: what the top lists, the treemap and the flow count, in one sentence
    assert E.NOTE_TOP["qty"] in d["notes"] and E.NOTE_TOP["value"] not in d["notes"]
    assert not any("are on MRP" in n for n in d["notes"])


def test_one_hospital_is_set_against_its_budget(client, data):
    d = ov(client, Month=AUG, Plant=DR)
    assert d["scope"]["plant"] == DR and d["scope"]["his_name"] == "HCG DR, Bangalore"
    k = kpis(d)
    # DR row 7: 3,000 / 2,000 against July's row 12, 500 / 450
    assert figures(k["sales"]) == (3000.0, 500.0, 500.0) and figures(k["margin_pct"]) == (33.33, 10.0, 23.33)
    assert figures(k["compliance"]) == (100.0, 0.0, 100.0)
    assert (k["compliance"]["target"], k["compliance"]["status"]) == (98.0, "met")
    assert "HCG's budget for this unit" in k["compliance"]["note"]
    # cash 680 + scheme 600 + out of formulary -1,600 (its brands earned more); July: scheme row 8, 80
    assert figures(k["losses"]) == (-320.0, 80.0, -500.0)
    # DR against its own usual cheapest: VIVITRA at 3,000 (1 of 7 vials), HERCLON 5 x (10,000 - 3,000);
    # against every hospital's (10,000) it paid nothing above, and the note says so. July: at its own
    assert figures(k["extra_paid"]) == (35000.0, 0.0, None)
    assert "it is ₹ 0." in k["extra_paid"]["note"]
    # the card says whose usual cheapest; the units table's row (DR: 0) is against all hospitals'
    assert k["extra_paid"]["label"] == "Paid above its own usual cheapest"
    assert k["extra_paid"]["note"].startswith("Paid above each product's usual cheapest price at this hospital")
    assert next(u["extra_paid"] for u in d["units"] if u["plant"] == DR) == 0.0
    ins = {i["id"]: i for i in d["insights"]}
    assert [i["id"] for i in d["insights"]] == ["cash_missed", "prices_extra", "sales_margin", "compliance",
                                                "oof_missed"]
    assert ins["compliance"]["text"] == ("Formulary compliance at HCG DR, Bangalore was 100.00% in Aug-26, against a "
                                         "budget of 98%: met. That is up 100.00 pts from Jul-26 (0.00%).")
    assert ins["oof_missed"]["text"] == ("The out-of-formulary brands bought earned ₹ 1,600 more margin than the "
                                         "suggested ones in Aug-26.")
    assert ins["oof_missed"]["tone"] == "up"
    assert ins["sales_margin"]["text"] == ("Revenue was ₹ 3,000 at MRP in Aug-26, up 500.00% from Jul-26, with a "
                                           "margin of 33.33% (10.00% in Jul-26).")    # no OP sales: no IP / OP sentence
    assert [(a["id"], a["impact"]) for a in d["actions"]] == [("negotiate_price", 35000.0), ("cash_switch", 680.0),
                                                              ("oof_switch", 600.0)]
    assert "₹3,000 a unit" in d["actions"][0]["text"]
    # KR: 15.79 pts short of 100%: an alert, and the follow-up names KR only
    k = ov(client, Month=AUG, Plant=KR)
    c = kpis(k)["compliance"]
    assert (c["value"], c["target"], c["status"]) == (84.21, 100.0, "below")
    assert k["insights"][0] == {"id": "compliance", "tone": "alert", "link": "/formulary?view=overview&month=2026-08",
                                "text": (
        "Formulary compliance at HCG KR, Bangalore was 84.21% in Aug-26, against a budget of 100%: 15.79 pts short. "
        "That is down 15.79 pts from Jul-26 (100.00%).")}
    assert [a["id"] for a in k["actions"]][-1] == "follow_up_units"


# ---------- months: the first, all of them, one source only ----------

def test_first_month_has_no_change_and_no_purchases(client, data):
    d = ov(client, Month=JUN)
    assert (d["scope"]["prev_month"], d["scope"]["prev_label"]) == (None, None)
    k = kpis(d)
    assert figures(k["sales"]) == (700.0, None, None) and figures(k["compliance"]) == (100.0, None, None)
    # June has sales but no spend and no purchases: those KPIs have no figure, never an error, and
    # their sparklines still show every month their sections hold
    assert figures(k["losses"]) == (None, None, None) and figures(k["extra_paid"]) == (None, None, None)
    assert spark(k["losses"]) == [("Jul-26", 3960.0), ("Aug-26", 7738.0)]
    assert spark(k["extra_paid"]) == [("Jul-26", 60000.0), ("Aug-26", 118000.0)]
    assert k["losses"]["note"] == "No part of what was missed has figures for this selection."
    assert k["extra_paid"]["note"].endswith("Prices paid on purchases have no figures for Jun-26 (they hold Jul-26 to "
                                            "Aug-26), so they are left out.")
    notes = d["notes"]
    assert "Jun-26 is the first month with sales, so no change is shown." in notes
    assert "Cash & TPA margin missed has no figures for Jun-26 (it holds Jul-26 to Aug-26), so it is left out." in notes
    assert ("Out-of-formulary purchases have no figures for Jun-26 (they hold Jul-26 to Aug-26), so they are left "
            "out.") in notes
    assert [u["oof_loss"] for u in d["units"]] == [None] * 3 and [u["extra_paid"] for u in d["units"]] == [None] * 3
    assert [u["delta_pts"] for u in d["units"]] == [None] * 3
    # only the margin is left in the bridge, and it says what is not
    assert [(s["key"], s["value"]) for s in d["bridge"]["steps"]] == [("earned", 300.0), ("possible", 300.0)]
    assert d["bridge"]["note"].endswith("Left out of the bridge, for want of figures for this selection: Cash & TPA, "
                                        "scheme, out of formulary.")
    # the rules without data leave themselves out: no losses, prices or switches to speak of
    assert [(i["id"], i["tone"]) for i in d["insights"]] == [("units_below_budget", "up"), ("sales_margin", "info"),
                                                             ("compliance", "info")]
    assert d["insights"][0]["text"] == "The only unit with a budget met it in Jun-26."
    assert d["insights"][1]["text"] == "Revenue was ₹ 700 at MRP in Jun-26, with a margin of 42.86%."
    assert d["actions"] == []


def test_all_months_are_totals_without_change(client, data):
    d = ov(client, Month="all")
    assert (d["scope"]["month"], d["scope"]["month_label"], d["scope"]["prev_month"]) == ("ALL", "All months", None)
    k = kpis(d)
    # sales June-August: 7,210 / 4,530 -> 37.17%; drug cost A 3,700, B 600, C 30 -> 86.05%, 13.86%
    assert [figures(k[x]) for x in ("sales", "margin_pct", "compliance", "oof_share")] == [
        (7210.0, None, None), (37.17, None, None), (86.05, None, None), (13.86, None, None)]
    # spend and purchases hold July and August: cash 4,140 + scheme 4,538 + out of formulary 3,020;
    # prices over both months at one usual cheapest (10,000): 18,000 + 1,00,000 + 40,000
    assert figures(k["losses"]) == (11698.0, None, None) and figures(k["extra_paid"]) == (158000.0, None, None)
    assert ("All months: every month on this page (Jun-26 to Aug-26). Each figure sums the months its section "
            "holds, amounts first and every % taken once, except extra paid, which is set against one usual "
            "cheapest over those months; no change is shown.") in d["notes"]
    assert "Cash & TPA margin missed covers Jul-26 to Aug-26 only: its figures are for those months." in d["notes"]
    assert "Prices paid on purchases cover Jul-26 to Aug-26 only: their figures are for those months." in d["notes"]
    # the extra paid is one usual cheapest over both months (1,58,000), not July's 60,000 + August's
    # 1,18,000 = 1,78,000 the sparkline adds up to, and its note says so
    assert sum(v for _, v in spark(k["extra_paid"])) == 178000.0
    assert k["extra_paid"]["note"].endswith("Over Jul-26 to Aug-26 it is set against one usual cheapest over all of "
                                            "them, so it is not the months' own figures (the sparkline) added up.")
    # what was missed: its three parts hold July and August, the sales June too; the bridge puts the
    # margin earned over three months beside parts over two, and says so instead of "same months"
    assert k["losses"]["note"] == ("Cash & TPA ₹ 4,140 + scheme ₹ 4,538 + out of formulary ₹ 3,020. Cash & TPA, "
                                   "scheme and out of formulary cover Jul-26 to Aug-26 only. Out of formulary is "
                                   "counted on purchases.")
    assert d["bridge"]["note"].endswith(
        "out of formulary on purchases (goods receipts), so the last bar is an indication, not one ledger. The months "
        "differ: margin earned covers Jun-26 to Aug-26; Cash & TPA, scheme and out of formulary cover Jul-26 to "
        "Aug-26.")
    assert "Out of formulary and extra paid are counted on purchases (goods receipts), not on sales." in d["notes"]
    # a link opens its page on All months; the formulary page's also holds May (the summary's)
    assert all(i["link"].endswith("month=ALL") for i in d["insights"])
    assert ("On its own page, All months for formulary compliance covers May-26 to Aug-26, more months than here, "
            "so its figures there differ.") in d["notes"]
    assert all(u["delta_pts"] is None for u in d["units"])
    # KR: A 1,300, B 150 -> 89.66%; DR: A 2,400, B 450 -> 84.21% against its 98%
    assert d["insights"][0]["text"] == ("2 of 2 units with a budget missed it from Jun-26 to Aug-26 (2 more than 2 pts "
                                        "short as HCG's MIS prints it). Furthest: HCG DR, Bangalore at 84.21% against "
                                        "98%, 13.79 pts short.")


def test_a_month_only_the_purchases_hold(client, data):
    # sales without July: the page still offers July (purchases hold it), sales and compliance have
    # no figure for it, and the change is against June, the previous month with sales
    rows = [r for r in SALES_ROWS if r[0] != JUL]
    si = sales_items_frame(rows)
    folder = data / "formulary"
    si.to_parquet(folder / "sales_items.parquet", index=False)
    si.groupby(SUMMARY_KEY, as_index=False)[["cost", "mrp", "qty", "lines"]].sum().to_parquet(
        folder / "formulary_summary.parquet", index=False)
    _clear()
    d = ov(client, Month=JUL)
    assert [(m["period"], m["sales"], m["purchases"]) for m in d["months"]] == [(JUN, True, False), (JUL, False, True),
                                                                               (AUG, True, True)]
    assert (d["scope"]["prev_month"], d["scope"]["prev_label"]) == (JUN, "Jun-26")
    k = kpis(d)
    assert figures(k["sales"]) == (None, None, None) and figures(k["compliance"]) == (None, None, None)
    assert spark(k["sales"]) == [("Jun-26", 700.0), ("Aug-26", 5050.0)]
    # July's cash 280 + scheme 1,880 + out of formulary 1,800; June is in no spend table: no change
    assert figures(k["losses"]) == (3960.0, None, None) and figures(k["extra_paid"]) == (60000.0, None, None)
    assert "Pharmacy sales have no figures for Jul-26 (they hold Jun-26, Aug-26), so they are left out." in d["notes"]
    assert d["bridge"]["steps"] == [] and d["ip_op"] == [] and d["mix"] is None
    assert d["treemap"] == [] and d["flow"] == {"nodes": [], "links": []}
    assert d["top"] == {"items": [], "units": [], "bottom": {"items": [], "units": []}}
    # no sales in July: the purchases are not "of the same months" as any sales
    assert "Out of formulary and extra paid are counted on purchases (goods receipts), not on sales." in d["notes"]
    kr = d["units"][0]
    assert (kr["mrp"], kr["compliance_pct"], kr["oof_loss"]) == (None, None, 1800.0)
    assert (kr["cash_loss"], kr["scheme_missed"], kr["extra_paid"]) == (280.0, 1800.0, 60000.0)
    assert [i["id"] for i in d["insights"]] == ["oof_missed", "cash_missed", "prices_extra"]
    assert d["insights"][0]["text"] == ("Out-of-formulary purchases missed ₹ 1,800 of margin in Jul-26. HCG KR, "
                                        "Bangalore missed the most: ₹ 1,800 (100.00% of the margin missed at units "
                                        "with a loss).")
    assert [(a["id"], a["impact"]) for a in d["actions"]] == [("negotiate_price", 60000.0), ("oof_switch", 1500.0),
                                                              ("cash_switch", 280.0)]


# ---------- an out-of-formulary month built without HCG's working, or waiting for its Item Master ----------
# The purchase build publishes a month without HCG's own worksheet from HCG's other months and
# lists (Estimated), and leaves out a month whose Item Master has not come (it waits). Neither
# changes a figure here: the losses KPI, the bridge and the notes say it beside the
# out-of-formulary part, in the words the Spend page's out-of-formulary view uses.
ESTIMATED_AUG = "Out of formulary: Estimated month (HCG's working for Aug-26 not received)."
WAITING_JUN = "Out of formulary for Jun-26 waits for its Item Master; prices and generic are built."


def _oof_without_hcgs_working(folder: Path, months: tuple) -> None:
    """oof_purchase as the fixture writes it, but `months` built without HCG's worksheet: their
    suggestions from HCG's sheet came from HCG's pair of another month (source 2) instead, so
    every figure stays the same and only where the suggestions came from changes."""
    rows = [r[:5] + (PO.HCG_SHEET_PAIR if r[0] in months and r[5] == PO.HCG_SHEET else r[5],) + r[6:]
            for r in OOF_ROWS]
    pd.DataFrame([oof_row(*r) for r in rows])[PO.OOF_COLUMNS].to_parquet(folder / f"{PC.OOF}.parquet", index=False)
    _clear()


def test_an_estimated_out_of_formulary_month_says_so(client, data):
    _oof_without_hcgs_working(data / "purchase", (AUG,))
    d = ov(client, Month=AUG)
    k = kpis(d)
    # the default month's figures, every one: 7,738 against July's 3,960 (see the first test)
    assert figures(k["losses"]) == (7738.0, 3960.0, 95.4)
    assert k["losses"]["note"] == ("Cash & TPA ₹ 3,860 + scheme ₹ 2,658 + out of formulary ₹ 1,220. Out of "
                                   "formulary is counted on purchases. " + ESTIMATED_AUG)
    assert [(s["key"], s["value"]) for s in d["bridge"]["steps"]][3] == ("oof", 1220.0)
    assert d["bridge"]["note"].endswith("so the last bar is an indication, not one ledger. " + ESTIMATED_AUG)
    assert ESTIMATED_AUG in d["notes"]
    # an estimated month is a property of the month: one hospital's selection says it too
    assert kpis(ov(client, Month=AUG, Plant=DR))["losses"]["note"].endswith(ESTIMATED_AUG)
    # July was built from HCG's working: nothing is said
    jul = ov(client, Month=JUL)
    assert "Estimated" not in json.dumps([kpis(jul)["losses"]["note"], jul["bridge"]["note"], jul["notes"]])
    # all months name the estimated one; two estimated months read as a span
    assert kpis(ov(client, Month="ALL"))["losses"]["note"].endswith(ESTIMATED_AUG)
    _oof_without_hcgs_working(data / "purchase", (JUL, AUG))
    every = ov(client, Month="ALL")
    both = "Out of formulary: Estimated months (HCG's working for Jul-26 to Aug-26 not received)."
    assert kpis(every)["losses"]["note"].endswith(both) and every["bridge"]["note"].endswith(both)
    # consumables leave out of formulary altogether, and say nothing about its months
    assert "Estimated" not in json.dumps(ov(client, Month=AUG, Segment="consumables"))


def test_a_month_waiting_for_its_item_master_says_so(client, data):
    # the purchase build's report: June's out-of-formulary purchases wait for June's Item Master and
    # oof_purchase has no June row, as a new month's would (/spend/meta oof_waiting lists it)
    path = data / "purchase" / PC.VALIDATION_FILE
    report = json.loads(path.read_text())
    report["months"] = {JUN: {"period": JUN, "oof": {"item_master": None, "waiting_for_item_master": True,
                                                     "warnings": [WAITING_JUN[:-1]]}}}
    path.write_text(json.dumps(report))
    _clear()
    d = ov(client, Month=JUN)
    k = kpis(d)
    assert figures(k["losses"]) == (None, None, None)
    assert k["losses"]["note"] == "No part of what was missed has figures for this selection. " + WAITING_JUN
    assert [s["key"] for s in d["bridge"]["steps"]] == ["earned", "possible"]
    assert d["bridge"]["note"].endswith("Left out of the bridge, for want of figures for this selection: Cash & TPA, "
                                        "scheme, out of formulary. " + WAITING_JUN)
    assert ("Out-of-formulary purchases have no figures for Jun-26 (they hold Jul-26 to Aug-26), so they are left "
            "out. " + WAITING_JUN) in d["notes"]
    # all months: out of formulary covers July and August, and says why June is not in it
    every = ov(client, Month="ALL")
    assert figures(kpis(every)["losses"]) == (11698.0, None, None)
    assert kpis(every)["losses"]["note"].endswith("Out of formulary is counted on purchases. " + WAITING_JUN)
    assert ("Out-of-formulary purchases cover Jul-26 to Aug-26 only: their figures are for those months. "
            + WAITING_JUN) in every["notes"]
    assert every["bridge"]["note"].endswith(WAITING_JUN)
    # a month that does not wait, and consumables (no out of formulary at all), say nothing of it
    for params in ({"Month": AUG}, {"Month": JUN, "Segment": "consumables"}):
        assert "waits for its Item Master" not in json.dumps(ov(client, **params), ensure_ascii=False), params


# ---------- filters, the units view, consumables, empty and unknown hospitals ----------

def test_units_view_counts_units_and_keeps_money_in_rupees(client, data):
    d = ov(client, Month=AUG, Basis="QTY")
    assert (d["scope"]["basis"], d["scope"]["budget_judged"]) == ("qty", False)
    k = kpis(d)
    # units sold: 109 against July's 29
    assert (k["sales"]["label"], k["sales"]["unit"]) == ("Units sold", "qty")
    assert figures(k["sales"]) == (109.0, 29.0, 275.86)
    # A 10 + 30 + 20, B 2, C 5 + 1: 60 / 62 = 96.77%; July A 8, B 5, C 6: 61.54%; no budget on units
    assert figures(k["compliance"]) == (96.77, 61.54, 35.23)
    assert (k["compliance"]["target"], k["compliance"]["status"]) == (None, None)
    assert figures(k["oof_share"]) == (2.94, 26.32, -23.38)
    # money stays money, and says so
    for key in ("margin_pct", "losses", "extra_paid"):
        assert "₹ only" in k[key]["note"], key
    assert (k["losses"]["unit"], k["losses"]["value"], k["extra_paid"]["value"]) == ("inr", 7738.0, 118000.0)
    kr = d["units"][0]
    # KR on units: A 40, B 2, C 5 -> 95.24%, shares of 47; July 100% (8 formulary units)
    assert (kr["compliance_pct"], kr["budget_pct"], kr["status"], kr["delta_pts"]) == (95.24, None, None, -4.76)
    assert (kr["formulary_share_pct"], kr["oof_share_pct"], kr["nf_share_pct"]) == (85.11, 4.26, 10.64)
    assert kr["mrp"] == 2050.0                                                   # the sales column stays MRP
    assert [x["share_pct"] for x in d["ip_op"]] == [66.97, 33.03]               # IP 73 of 109 units
    assert [(t["key"], t["value"]) for t in d["treemap"]] == [("onco", 33.0), ("non-onco", 36.0), ("consumables", 40.0)]
    assert [(i["material"], i["value"]) for i in d["top"]["items"]][:3] == [("300001", 40.0), ("100001", 30.0),
                                                                             ("200001", 30.0)]
    assert d["mix"]["total"] == 68.0 and "The bridge is in rupees" in d["bridge"]["note"]
    assert any(n.startswith("Quantity view:") for n in d["notes"])


def test_consumables_have_no_compliance_and_no_purchases(client, data):
    d = ov(client, Month=AUG, Segment="consumables")
    k = kpis(d)
    # row 5: 400 / 100 -> 75%; July row 10: 100 / 60
    assert figures(k["sales"]) == (400.0, 100.0, 300.0) and k["margin_pct"]["value"] == 75.0
    assert [k[x]["value"] for x in ("compliance", "oof_share", "extra_paid")] == [None, None, None]
    # no consumable in the cash list; the scheme's J&J suture (row 4) misses 28 - 20 = 8
    assert k["losses"]["value"] == 8.0
    assert k["losses"]["note"] == ("Cash & TPA ₹ 0 + scheme ₹ 8. Not included, for want of figures for this selection: "
                                   "out of formulary.")
    assert d["mix"] is None and d["scope"]["budget_judged"] is False
    assert E.NOTE_CONSUMABLES in d["notes"] and E.NOTE_BUDGET_FILTERS in d["notes"]
    assert all(u["compliance_pct"] is None and u["oof_loss"] is None and u["extra_paid"] is None for u in d["units"])
    assert [(t["key"], t["value"]) for t in d["treemap"]] == [("consumables", 400.0)]


def test_patient_type_cuts_sales_but_not_purchases(client, data):
    d = ov(client, Month=AUG, Type="op")
    k = kpis(d)
    # OP rows 3, 4, 8: 400 / 200; purchases have no patient type
    assert (k["sales"]["value"], k["extra_paid"]["value"], d["scope"]["type"]) == (400.0, 118000.0, "OP")
    assert any("the IP/OP filter does not apply to them" in n for n in d["notes"])
    assert d["scope"]["budget_judged"] is False and all(u["status"] is None for u in d["units"])
    # the sales card says whose sales it counts, in rupees and in units
    assert k["sales"]["note"].startswith("Total MRP of the HIS pharmacy sales lines of OP patients: that part of "
                                         "HCG's revenue.")
    assert kpis(ov(client, Month=AUG, Type="OP", Basis="qty"))["sales"]["note"].startswith(
        "Units sold on the HIS pharmacy sales lines of OP patients: tablets, vials")
    assert kpis(ov(client, Month=AUG, Segment="consumables"))["sales"]["note"].startswith(
        "Total MRP of the HIS pharmacy sales lines, IP and OP, Consumables only: that part of HCG's revenue.")


def test_links_carry_the_month_and_the_filters_each_page_takes(client, data):
    d = ov(client, Month=AUG, Type="IP", Segment="onco")
    links = {x["id"]: parse_qs(urlsplit(x["link"]).query) for x in d["insights"] + d["actions"]}
    assert {"sales_margin", "compliance", "cash_missed", "oof_missed", "prices_extra"} <= set(links)
    for key in ("sales_margin", "compliance", "cash_missed"):              # sales pages: month, type, segment
        assert (links[key]["month"], links[key]["type"], links[key]["segment"]) == ([AUG], ["IP"], ["onco"]), key
    for key in ("oof_missed", "prices_extra"):                               # purchases: no patient type
        assert (links[key]["month"], links[key]["segment"], "type" in links[key]) == ([AUG], ["onco"], False), key
    assert links["compliance"]["view"] == ["overview"] and links["oof_missed"]["section"] == ["oof"]


def _budgets(folder: Path, rows: list) -> None:
    """hcg_mis_compliance with (plant, period, budget %) rows."""
    pd.DataFrame([{"plant": p, "period": m, "budget_pct": b, "hcg_actual_pct": b, "hcg_formula": "pooled",
                   "pooled_pct": b} for p, m, b in rows]).to_parquet(folder / "hcg_mis_compliance.parquet", index=False)
    _clear()


BUDGETS_DIFFER = ("HCG's MIS sets this unit 100% in Jun-26, Aug-26 and 95% in Jul-26, so the months selected are "
                  "not set against one budget.")


def test_budgets_that_differ_across_the_months_say_so(client, data):
    # KR's budget is 95% in July and 100% in August (June takes the FY's latest, August's 100%):
    # all months have a budget, but no one budget to set the span against
    _budgets(data / "formulary", [(KR, JUL, 95.0), (KR, AUG, 100.0), (DR, "2026-04", 98.0), (RAJKOT, "2026-04", 85.0)])
    assert formulary._budget(KR, [JUL, AUG]) == (None, formulary.BUDGET_DIFFERS)
    assert formulary._budget(KR, [AUG]) == (100.0, "hcg_mis") and formulary._budget(GOLKULA, [AUG]) == (None, None)
    d = ov(client, Month="ALL", Plant=KR)
    c = kpis(d)["compliance"]
    assert (c["target"], c["status"]) == (None, None) and c["note"].endswith(BUDGETS_DIFFER)
    kr, dr = (next(u for u in d["units"] if u["plant"] == p) for p in (KR, DR))
    assert (kr["budget_pct"], kr["status"], kr["budget_note"]) == (None, None, BUDGETS_DIFFER)
    assert (dr["budget_pct"], dr["budget_note"]) == (98.0, None)
    # one month has its one budget
    assert kpis(ov(client, Month=AUG, Plant=KR))["compliance"]["target"] == 100.0
    # the formulary page says it too, beside a budget_source of its own
    f = client.get("/formulary/overview", params={"Plant": KR, "Month": f"{JUN},{JUL},{AUG}"}).json()
    assert (f["headline"]["budget_pct"], f["headline"]["budget_source"]) == (None, "differs")
    assert BUDGETS_DIFFER in f["notes"]


def _ctx(compliance: float, budget: float) -> E.Ctx:
    """One unit's compliance against its budget, as the two insight rules read it."""
    status = E.classify(compliance, budget)
    unit = {"plant": KR, "name": "KR", "his_name": "HCG KR, Bangalore", "compliance_pct": compliance,
            "budget_pct": budget, "status": status, "selected": False}
    other = {**unit, "plant": DR, "his_name": "HCG DR, Bangalore", "compliance_pct": 99.0, "budget_pct": 98.0,
             "status": "met"}
    k = {"compliance": {"value": compliance, "target": budget, "status": status, "prev": None, "delta": None}}
    src = {"formulary": E.Part({"headline": {}}, (AUG,), None, None)}
    return E.Ctx(E.Scope(None, AUG, None, None, "value"), src, k, [unit, other], "across all hospitals", None, True)


def test_within_is_said_on_the_figure_hcg_prints():
    # 89.60% against 92%: 2.40 pts short, but HCG's MIS prints 90%, 2 pts below: within (the card's
    # rule). The words say it is the printed figure, and the "more than 2 pts" count follows the chips.
    c = _ctx(89.6, 92.0)
    assert E._insight_compliance(c)["text"] == ("Formulary compliance across all hospitals was 89.60% in Aug-26, "
                                                "against a budget of 92%: 2.40 pts short; HCG's MIS prints it within "
                                                "2 pts of the budget.")
    assert E._insight_units_below(c)["text"] == (
        "1 of 2 units with a budget missed it in Aug-26 (0 more than 2 pts short as HCG's MIS prints it). Furthest: "
        "HCG KR, Bangalore at 89.60% against 92%, 2.40 pts short.")
    # 89.49% prints 89%, 3 pts below: below, and counted
    assert "(1 more than 2 pts short as HCG's MIS prints it)" in E._insight_units_below(_ctx(89.49, 92.0))["text"]
    assert E._insight_compliance(_ctx(89.49, 92.0))["text"].endswith("against a budget of 92%: 2.51 pts short.")


def test_parts_of_other_months_are_named_beside_the_figure(client, data):
    # Out of formulary built for August only, the spend tables July and August, sales June to
    # August: under all months the losses add parts of two windows, and the bridge sets them
    # beside a margin earned over three months. Both say which months each part covers, and the
    # purchases are no longer "of the same months" as the sales.
    rows = [r for r in OOF_ROWS if r[0] == AUG]
    pd.DataFrame([oof_row(*r) for r in rows])[PO.OOF_COLUMNS].to_parquet(data / "purchase" / f"{PC.OOF}.parquet",
                                                                         index=False)
    _clear()
    d = ov(client, Month="ALL")
    k = kpis(d)
    # cash 280 + 3,860 = 4,140; scheme 1,880 + 2,658 = 4,538; out of formulary August's 1,220
    assert k["losses"]["value"] == 9898.0
    assert k["losses"]["note"] == ("Cash & TPA ₹ 4,140 + scheme ₹ 4,538 + out of formulary ₹ 1,220. The months "
                                   "differ: Cash & TPA and scheme cover Jul-26 to Aug-26; out of formulary covers "
                                   "Aug-26. Out of formulary is counted on purchases.")
    assert d["bridge"]["note"].endswith("purchases (goods receipts), so the last bar is an indication, not one ledger. "
                                        "The months differ: margin earned covers Jun-26 to Aug-26; Cash & TPA and "
                                        "scheme cover Jul-26 to Aug-26; out of formulary covers Aug-26.")
    assert "Out of formulary and extra paid are counted on purchases (goods receipts), not on sales." in d["notes"]
    # one month: every part holds it, nothing to say
    one = ov(client, Month=AUG)
    assert "months differ" not in kpis(one)["losses"]["note"] + one["bridge"]["note"]
    assert "of the same months" in one["bridge"]["note"]


def test_a_hospital_with_no_rows_answers_with_zeros(client, data):
    other = next(c for c in sorted(da.load("dim_plant")["plant"].astype(str)) if c not in (KR, DR, RAJKOT, GOLKULA))
    d = ov(client, Month=AUG, Plant=other)
    k = kpis(d)
    assert (d["scope"]["plant"], k["sales"]["value"], k["sales"]["prev"], k["sales"]["delta"]) == (other, 0.0, 0.0,
                                                                                                    None)
    assert (k["margin_pct"]["value"], k["compliance"]["value"], k["losses"]["value"], k["extra_paid"]["value"]) == \
        (None, None, 0.0, 0.0)
    assert any(n.startswith("No pharmacy sales for") for n in d["notes"])
    assert d["flow"] == {"nodes": [], "links": []} and d["top"]["items"] == [] and len(d["top"]["units"]) == 2
    assert [i["id"] for i in d["insights"]] == ["sales_margin", "oof_missed", "cash_missed"]
    assert d["insights"][0]["text"] == "Revenue was ₹ 0 at MRP in Aug-26, about the same as Jul-26."
    assert d["actions"] == []


def test_bad_arguments_and_an_unknown_hospital(client, data):
    for params, status in [
        ({"Plant": "NOWHERE"}, 404),
        ({"Basis": "cost"}, 400),
        ({"Type": "ER"}, 400),
        ({"Segment": "drugs"}, 400),
        ({"Month": "Aug-26"}, 400),
        ({"Month": f"{JUL},{AUG}"}, 400),                   # one month or ALL
        ({"Month": MAY}, 400),                              # in the formulary summary only: no page month
        ({"Month": "2026-09"}, 400),
    ]:
        r = client.get("/executive/overview", params=params)
        assert r.status_code == status, (params, r.text)


def test_needs_a_login(anon_client, data):
    assert anon_client.get("/executive/overview").status_code == 401


# ---------- sections not built ----------

def test_spend_not_built_leaves_its_parts_out(client, data, monkeypatch):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(data / "spend-not-built"))
    _clear()
    d = ov(client, Month=AUG)
    k = kpis(d)
    # only the out-of-formulary purchases are left of what was missed: 1,220 against July's 1,800
    assert figures(k["losses"]) == (1220.0, 1800.0, -32.22)
    assert k["losses"]["note"].endswith("Not included, for want of figures for this selection: Cash & TPA, scheme.")
    assert "Cash & TPA margin missed is not built on this server yet, so it is left out." in d["notes"]
    assert [u["cash_loss"] for u in d["units"]] == [None] * 3 and d["units"][0]["oof_loss"] == 2820.0
    assert [s["key"] for s in d["bridge"]["steps"]] == ["earned", "oof", "possible"]
    assert "cash_missed" not in [i["id"] for i in d["insights"]]
    assert "cash_switch" not in [a["id"] for a in d["actions"]]
    # and the purchases too: nothing missed has a figure, and the page offers the sales months only
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(data / "purchase-not-built"))
    _clear()
    d = ov(client, Month=AUG)
    k = kpis(d)
    assert (k["losses"]["value"], k["extra_paid"]["value"], k["losses"]["spark"]) == (None, None, [])
    assert [m["purchases"] for m in d["months"]] == [False, False, False]
    assert "Out-of-formulary purchases are not built on this server yet, so they are left out." in d["notes"]
    assert [s["key"] for s in d["bridge"]["steps"]] == ["earned", "possible"]
    assert [i["id"] for i in d["insights"]] == ["units_below_budget", "sales_margin", "compliance"]
    assert [a["id"] for a in d["actions"]] == ["follow_up_units"]


def test_sales_not_built_is_503(client, data):
    (data / "formulary" / "sales_items.parquet").unlink()
    _clear()
    r = client.get("/executive/overview")
    assert r.status_code == 503 and "--from-cache" in r.json()["detail"]


def test_a_rebuild_is_picked_up_without_a_restart(client, data):
    assert kpis(ov(client, Month=AUG))["losses"]["value"] == 7738.0
    (data / "purchase" / "oof_purchase.parquet").unlink()       # the purchase build's table is gone
    assert kpis(ov(client, Month=AUG))["losses"]["value"] == 6518.0       # cash 3,860 + scheme 2,658


# ---------- HCG's number style and the status rule (ported from the UI) ----------

@pytest.mark.parametrize("v,text", [
    (3243199.13, "₹ 32.43 L"), (943122638.17, "₹ 9,431 L"), (54642940.44, "₹ 546 L"), (99999.6, "₹ 1.00 L"),
    (9999999.0, "₹ 100 L"), (31030.4, "₹ 31,030"), (1012500.0, "₹ 10.13 L"), (-1600.0, "−₹ 1,600"), (-0.4, "₹ 0"),
    (0.0, "₹ 0"), (None, "—"), (float("nan"), "—"),
])
def test_lakhs_prints_as_the_ui_does(v, text):
    assert E.lakhs(v) == text


def test_other_number_words():
    assert (E.unit_price(25599.0), E.unit_price(18938.1), E.unit_price(0.76)) == ("₹25,599", "₹18,938.10", "₹0.76")
    assert (E.pct(97.556), E.pct(-1.5)) == ("97.56%", "−1.50%")
    assert (E.pts(15.786), E.budget_text(97.5), E.budget_text(100.0)) == ("15.79 pts", "97.5%", "100%")
    assert (E._indian(12345678), E._indian(999), E.units_text(4490262.0)) == ("1,23,45,678", "999", "44,90,262 units")
    assert (E.units_text(1.2), E.units_text(0.0), E.units_text(None)) == ("1 unit", "0 units", "—")
    # the extra-paid note's share: the prices answer's own rule, else the Prices page's (never a literal)
    assert (E._usual_share({"usual_share_pct": 12.5}), E._usual_share({"usual_share_pct": 12.345})) == ("12.5%",
                                                                                                    "12.35%")
    assert E._usual_share(None) == E._usual_share({"usual_share_pct": None}) == f"{PP.USUAL_SHARE * 100:g}%"
    assert E.product_name({"minor_group": "CEFTAZIDIME+AVIBACTAM", "strength": "0.5MG|2000MG", "form": "INJ",
                           "kind": None}) == "CEFTAZIDIME+AVIBACTAM 0.5MG + 2000MG INJ"


@pytest.mark.parametrize("value,budget,status", [
    (97.5, 98.0, "met"), (97.49, 98.0, "within"), (95.5, 98.0, "within"), (95.49, 98.0, "below"),
    (100.0, 100.0, "met"), (None, 98.0, None), (97.0, None, None),
])
def test_status_is_the_cards_rule(value, budget, status):
    assert E.classify(value, budget) == status


# ---------- the committed tables: Aug-26, every section's own figure ----------

_DATA = {"sales": Path(settings.FORMULARY_DATA_DIR) / "sales_items.parquet",
         "spend": Path(settings.SPEND_DATA_DIR) / "cash_switch.parquet",
         "oof": Path(settings.PURCHASE_DATA_DIR) / "oof_purchase.parquet",
         "prices": Path(settings.PURCHASE_DATA_DIR) / "purchase_prices.parquet"}
_real = pytest.mark.skipif(not all(p.exists() for p in _DATA.values()),
                           reason="sales, spend or purchase tables not built")


def _get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


@_real
def test_real_aug_26_matches_every_section(client):
    _clear()
    if AUG not in set(pd.read_parquet(_DATA["sales"], columns=["period"])["period"].astype(str)):
        pytest.skip(f"sales_items has no {AUG}")
    d = ov(client, Month=AUG)
    k = kpis(d)
    assert [x["key"] for x in d["kpis"]] == KPI_KEYS
    rev = _get(client, "/revenue/overview", Month=AUG)
    form = _get(client, "/formulary/overview", Month=AUG)
    cash = _get(client, "/spend/cash/overview", Month=AUG)
    scheme = _get(client, "/spend/scheme/overview", Month=AUG)
    oof = _get(client, "/spend/oof/overview", Month=AUG)
    prices = _get(client, "/spend/prices/overview", Month=AUG)
    # HCG's Aug-26 consolidated sales file and its out-of-formulary Summary
    assert k["sales"]["value"] == rev["headline"]["mrp"] == 943_122_638.17
    assert k["sales"]["prev"] == rev["prev"]["mrp"] and k["margin_pct"]["value"] == rev["headline"]["margin_pct"]
    assert k["compliance"]["value"] == round(form["headline"]["pooled_pct"], 2)
    h = form["headline"]
    assert k["oof_share"]["value"] == round(h["out_of_formulary"] / (h["formulary"] + h["out_of_formulary"]
                                                                    + h["non_formulary"]) * 100, 2)
    assert oof["headline"]["margin_loss"] == 3_243_199.13
    assert k["losses"]["value"] == round(cash["headline"]["margin_loss"] + scheme["headline"]["saving_missed"]
                                         + oof["headline"]["margin_loss"], 2)
    assert k["extra_paid"]["value"] == prices["headline"]["extra_paid_vs_cheapest"]
    steps = {s["key"]: s["value"] for s in d["bridge"]["steps"]}
    assert (steps["earned"], steps["oof"]) == (rev["headline"]["margin"], 3_243_199.13)
    # HCG's 22 PAN units and Ratlam (HO02), which sells but is not in the list: 23. MSR-Golkula
    # (HC03), named only in HCG's history sheets (in_pan False) and with no sales, compliance or
    # purchase figure in any month, is left out, as Plant=HC03 is unknown (RV1-09)
    units = {u["plant"]: u for u in d["units"]}
    pan = {p for p, m in E._mis_units().items() if m["in_pan"]}
    assert len(pan) == 22 and set(units) == pan | {u["plant"] for u in rev["by_hospital"]}
    assert len(units) == 23 and "HO02" in units and "HO02" not in pan and "HC03" not in units
    assert client.get("/executive/overview", params={"Plant": "HC03"}).status_code == 404
    # D6: the blank-cost lines the Revenue page reports, beside the margin
    blank = rev["headline"]["cost_missing_lines"]
    assert k["margin_pct"]["tag"] == (f"incl. {E._indian(blank)} lines with no cost" if blank else None)
    # every insight and action opens its page on August
    assert all(x["link"].endswith("month=2026-08") or x["link"].endswith("month=2026-08&check=1")
               for x in d["insights"] + d["actions"])
    assert units["HC05"]["oof_loss"] == next(u["margin_loss"] for u in oof["by_hospital"] if u["plant"] == "HC05")
    assert units["HC05"]["compliance_pct"] == round(next(
        u["compliance_pct"] for u in form["by_hospital"] if u["plant"] == "HC05"), 2)
    assert 1 <= len(d["insights"]) <= 6 and 1 <= len(d["actions"]) <= 4
    assert all(a["impact"] is None or a["impact"] > 0 for a in d["actions"])
    assert json.dumps(d, allow_nan=False)
    # every month alike: the units view and all months answer too
    assert kpis(ov(client, Month=AUG, Basis="qty"))["compliance"]["target"] is None
    assert kpis(ov(client, Month="ALL"))["sales"]["delta"] is None
