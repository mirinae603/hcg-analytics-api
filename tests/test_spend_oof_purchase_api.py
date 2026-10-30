# tests/test_spend_oof_purchase_api.py — the out-of-formulary views on PURCHASES (HCG's GRN
# method): /spend/oof/overview|items|vendors, the cash overview's `with_oof`, the Brand filter's
# out-of-formulary side and /spend/meta, served from the purchase build's tables.
#
# The tables are written here from 11 invented oof_purchase rows (already aggregated, as
# `python -m app.etl.purchase` publishes them) and an HCG Summary for Aug only, beside the
# spend fixture of tests/test_spend_api.py. Every expected number was worked out by hand from
# them (comments show the arithmetic). The last tests run on the COMMITTED tables and set the
# API's figures beside HCG's own Summary and spend email.
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.api import spend
from app.core.config import settings
from app.etl.purchase import config as PC
from app.etl.purchase import oof as PO
from app.etl.purchase.masters import SUMMARY_COLUMNS

from tests.test_formulary_api import write_hospital_map
from tests.test_spend_api import AUG, JUL, get, write_tables

KR, DR, HC13 = "HC05", "HC06", "HC13"
V1, V2 = "50001", "50002"
VENDORS = {V1: "Alpha Pharma", V2: "Beta Distributors"}
# A vendor is the supplier by name (spacing and case aside), as the vendor views key it.
ALPHA, BETA = "ALPHA PHARMA", "BETA DISTRIBUTORS"
# OOF brand -> (name, molecule, maker)
BRANDS = {"101100": ("ONCO OOF A 100MG INJ", "MOLECULE A", "MAKER ONE"),
          "201200": ("PARA OOF B TAB", "PARACETAMOL", "MAKER TWO"),
          "201300": ("BETTER C TAB", "MOLECULE C", "MAKER TWO"),
          "201400": ("NOALT D INJ", "MOLECULE D", "MAKER ONE"),
          "201500": ("PARKED E TAB", "MOLECULE E", "MAKER ONE"),
          "101600": ("RULE F 50MG INJ", "MOLECULE F", "MAKER THREE"),
          "201700": ("UNPRICED G SYRUP", "MOLECULE G", "MAKER THREE")}
SUGGESTED = {"101900": "ONCO FORM A 100MG INJ", "201900": "PARA FORM B TAB", "201901": "FORM C TAB",
             "101901": "FORM F 50MG INJ", "201902": "FORM G SYRUP"}
# GST 5% throughout, so MRP 1050 is 1000 without GST. Columns: period, plant, brand, vendor,
# status, source, rule level, suggested brand, price source, lines, qty, Base Price, unit MRP,
# suggested price, suggested MRP. Per row: PP = Base Price x qty; margin earned = qty x (MRP
# excl GST - Base Price); possible = qty x (suggested MRP excl GST - suggested price).
ROWS = [
    # r1  6000 PP | earned 10 x (1000 - 600) = 4000 | possible 10 x (1000 - 300) = 7000 | missed 3000
    (AUG, KR, "101100", V1, PO.COUNTED, PO.HCG_SHEET, None, "101900", PO.PRICE_HCG_SHEET, 2, 10, 600, 1050, 300, 1050),
    # r2  1000 PP | 20 x (100 - 50) = 1000 | 20 x (100 - 20) = 1600 | 600
    (AUG, KR, "201200", V2, PO.COUNTED, PO.HCG_SHEET, None, "201900", PO.PRICE_HCG_SHEET, 1, 20, 50, 105, 20, 105),
    # r3  400 PP | 10 x (200 - 40) = 1600 | 10 x (200 - 150) = 500 | -1100: the brand bought earns more
    (AUG, KR, "201300", V2, PO.COUNTED, PO.HCG_SHEET, None, "201901", PO.PRICE_HCG_SHEET, 1, 10, 40, 210, 150, 210),
    # r4  500 PP, no alternative mapped: shown, adds nothing
    (AUG, KR, "201400", V1, PO.NO_ALTERNATIVE, PO.NONE, None, None, None, 1, 5, 100, 157.5, None, None),
    # r5  300 PP, parked by HCG: shown, adds nothing
    (AUG, KR, "201500", V1, PO.PARKED, PO.HCG_SHEET, None, None, None, 1, 10, 30, 63, None, None),
    # r6  800 PP | 4 x (300 - 200) = 400 | 4 x (300 - 120) = 720 | 320, the app's rule (Estimated)
    (AUG, KR, "101600", V2, PO.COUNTED, PO.FORMULARY_RULE, "A" + PO.FUZZY_MARK, "101901", PO.PRICE_CASH_TPA, 1, 4,
     200, 315, 120, 315),
    # r7  100 PP, a suggestion no price list prices: counted as no alternative
    (AUG, KR, "201700", V2, PO.UNPRICED, PO.HCG_SHEET_PAIR, None, "201902", None, 1, 10, 10, 21, None, None),
    # r8  1200 PP | 2 x 400 = 800 | 2 x 700 = 1400 | 600
    (AUG, DR, "101100", V1, PO.COUNTED, PO.HCG_SHEET, None, "101900", PO.PRICE_HCG_SHEET, 1, 2, 600, 1050, 300, 1050),
    # r9  800 PP | 20 x 160 = 3200 | 20 x 50 = 1000 | -2200: DR's loss is negative (600 - 2200)
    (AUG, DR, "201300", V2, PO.COUNTED, PO.HCG_SHEET, None, "201901", PO.PRICE_HCG_SHEET, 1, 20, 40, 210, 150, 210),
    # r10 3000 PP | 5 x 400 = 2000 | 5 x 700 = 3500 | 1500
    (JUL, KR, "101100", V1, PO.COUNTED, PO.HCG_SHEET, None, "101900", PO.PRICE_HCG_SHEET, 1, 5, 600, 1050, 300, 1050),
    # r11 500 PP | 10 x 50 = 500 | 10 x 80 = 800 | 300, from HCG's master this time
    (JUL, KR, "201200", V2, PO.COUNTED, PO.HCG_LIST, None, "201900", PO.PRICE_LIST_ROW, 1, 10, 50, 105, 20, 105),
]
GST = 5.0
# HCG's own Summary covers Aug only: KR and DR as ours, plus a centre only HCG lists.
HCG_SUMMARY = [(AUG, KR, "KR", 8200.0, 15200.0, 7000.0, 5380.0, 15200.0, 9820.0, 2820.0),
               (AUG, DR, "DR", 2000.0, 6000.0, 4000.0, 3600.0, 6000.0, 2400.0, -1600.0),
               (AUG, HC13, "HCG only", 100.0, 150.0, 50.0, 50.0, 150.0, 100.0, 50.0)]
BUILT_AT = "2026-10-20T10:00:00+00:00"


def _row(period, plant, material, vendor, status, source, level, sug, price_source, lines, qty, base, mrp, sp, sm):
    name, molecule, maker = BRANDS[material]
    abb = mrp / (100 + GST) * 100
    pp, total_mrp = base * qty, abb * qty
    counted = status == PO.COUNTED
    sug_pp = sp * qty if counted else 0.0
    sug_mrp = sm / (100 + GST) * 100 * qty if counted else 0.0
    actual = total_mrp - pp if counted else 0.0
    loss = (sug_mrp - sug_pp) - actual if counted else 0.0
    return {"period": period, "plant": plant, "category": "Onco Drugs" if material[0] == "1" else "Other Drugs",
            "segment": "onco" if material[0] == "1" else "non-onco", "material": material, "material_desc": name,
            "minor_group": molecule, "manufacturer": maker, "vendor_code": vendor, "vendor_name": VENDORS[vendor],
            "status": status, "suggestion_source": source, "rule_level": level, "suggested_material": sug,
            "suggested_desc": SUGGESTED.get(sug), "price_source": price_source, "sug_price": sp, "sug_mrp": sm,
            "sug_gst": GST if sp is not None else None, "base_price_min": float(base), "base_price_max": float(base),
            "lines": lines, "qty": float(qty), "total_pp": float(pp), "total_wo_tax": 0.9 * pp,
            "total_mrp": total_mrp, "actual_margin": actual, "sug_total_pp": sug_pp, "sug_total_mrp": sug_mrp,
            "suggested_margin": sug_mrp - sug_pp, "margin_loss": loss, "gain_loss": max(loss, 0.0),
            "better_loss": min(loss, 0.0)}


def write_oof_tables(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame([_row(*r) for r in ROWS])[PO.OOF_COLUMNS]
    oof.to_parquet(folder / f"{PC.OOF}.parquet", index=False)
    pd.DataFrame(HCG_SUMMARY, columns=SUMMARY_COLUMNS).to_parquet(folder / f"{PC.HCG_SUMMARY}.parquet", index=False)
    (folder / PC.VALIDATION_FILE).write_text(json.dumps({"built_at": BUILT_AT, "oof": {"months": [JUL, AUG]}}))


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "spend"))
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase"))
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary-not-built"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "spend")
    write_oof_tables(tmp_path / "purchase")
    write_hospital_map(tmp_path / "reference")
    spend.clear_caches()
    yield tmp_path
    spend.clear_caches()


def oof(client, **params):
    return get(client, "/spend/oof/overview", **params)


def margins(b) -> tuple:
    return b["actual_margin"], b["potential_margin"], b["margin_loss"]


# ---------- headline ----------

def test_headline_is_hcgs_net_sum(client, data):
    h = oof(client, Plant=KR, Month=AUG)["headline"]
    # rows 1-7: every line's PP 6000 + 1000 + 400 + 500 + 300 + 800 + 100; HCG's used PP the counted 1-3, 6
    assert (h["purchase_value"], h["counted_value"], h["value_after_discount"]) == (9100.0, 8200.0, 8190.0)
    # earned 4000 + 1000 + 1600 + 400; possible 7000 + 1600 + 500 + 720; missed NET 3000 + 600 - 1100 + 320
    assert margins(h) == (7000.0, 9820.0, 2820.0)
    assert (h["gain_loss"], h["better_loss"]) == (3920.0, -1100.0)
    assert h["loss_pct"] == pytest.approx(2820 / 9820 * 100, abs=1e-4)
    assert (h["suggested_value"], h["used_mrp_value"], h["suggested_mrp_value"]) == (5380.0, 15200.0, 15200.0)
    assert h["used_margin_pct"] == pytest.approx(7000 / 15200 * 100, abs=1e-4)
    assert (h["qty"], h["lines"], h["brands"], h["vendors"], h["hospitals"]) == (69.0, 8, 7, 2, 1)


def test_coverage_says_where_each_suggestion_came_from(client, data):
    cov = oof(client, Plant=KR, Month=AUG)["headline"]["coverage"]
    assert list(cov) == [PO.HCG_SHEET, PO.HCG_SHEET_PAIR, PO.HCG_LIST, PO.FORMULARY_RULE, PO.PARKED, PO.NONE]
    got = {k: (v["value"], v["lines"], v["brands"]) for k, v in cov.items()}
    # the unpriced pair (row 7) has no alternative, like row 4; parked row 5 apart
    assert got == {PO.HCG_SHEET: (7400.0, 4, 3), PO.HCG_SHEET_PAIR: (0.0, 0, 0), PO.HCG_LIST: (0.0, 0, 0),
                   PO.FORMULARY_RULE: (800.0, 1, 1), PO.PARKED: (300.0, 1, 1), PO.NONE: (600.0, 2, 2)}
    assert sum(v["value"] for v in cov.values()) == 9100.0
    assert cov[PO.HCG_SHEET]["share_pct"] == pytest.approx(7400 / 9100 * 100, abs=1e-4)
    assert [k for k, v in cov.items() if v["estimated"]] == [PO.FORMULARY_RULE]
    assert [k for k, v in cov.items() if not v["counted"]] == [PO.PARKED, PO.NONE]


def test_all_hospitals_and_several_months_pool_the_sums(client, data):
    assert margins(oof(client, Month=AUG)["headline"]) == (11000.0, 12220.0, 1220.0)     # + DR 4000 / 2400 / -1600
    d = oof(client, Plant=KR, Month="ALL")
    h = d["headline"]
    assert margins(h) == (9500.0, 14120.0, 4620.0)                                     # + Jul 2500 / 4300 / 1800
    assert h["loss_pct"] == pytest.approx(4620 / 14120 * 100, abs=1e-4)
    assert any("never an average" in n for n in d["notes"])


def test_month_is_the_purchase_builds(client, data):
    assert oof(client, Plant=KR)["scope"]["months"] == [AUG]                           # latest OOF month
    r = client.get("/spend/oof/overview", params={"Month": "2026-06"})
    assert r.status_code == 400 and "No out-of-formulary purchase data for 2026-06" in r.json()["detail"]


def test_type_and_payer_are_accepted_and_not_applied(client, data):
    plain = oof(client, Plant=KR, Month=AUG)
    d = oof(client, Plant=KR, Month=AUG, Type="IP", Payer="CASH(TPA)")
    assert d["headline"] == plain["headline"]
    assert d["scope"]["not_applied"] == ["Type", "Payer"] and "type" not in d["scope"]
    assert any("Type and Payer filters do not apply" in n for n in d["notes"])
    assert plain["scope"]["not_applied"] == []
    assert not any("do not apply" in n or "does not apply" in n for n in plain["notes"])
    for params in ({"Type": "XX"}, {"Payer": "NOBODY"}, {"Segment": "bones"}):
        assert client.get("/spend/oof/overview", params=params).status_code == 400, params
    assert client.get("/spend/oof/overview", params={"Plant": "NOWHERE"}).status_code == 404


def test_segment_and_brand_cut_the_purchases(client, data):
    assert margins(oof(client, Plant=KR, Month=AUG, Segment="onco")["headline"]) == (4400.0, 7720.0, 3320.0)
    assert oof(client, Plant=KR, Month=AUG, Category="Other Drugs")["headline"]["margin_loss"] == -500.0
    b = oof(client, Month=AUG, Material="101100.0")
    assert (b["scope"]["material"], b["headline"]["margin_loss"], b["headline"]["brands"]) == ("101100", 3600.0, 1)

def test_consumables_are_not_in_the_purchases(client, data):
    # HCG's purchase scope keeps the drug series only, so the OOF views refuse consumables, as
    # the Prices and Generic vs branded views do; the cash overview still takes them
    for path in ("/spend/oof/overview", "/spend/oof/items", "/spend/oof/vendors"):
        for params in ({"Segment": "consumables"}, {"Category": "Consumables"}):
            r = client.get(path, params={"Plant": KR, "Month": AUG, **params})
            assert r.status_code == 400 and "Consumables are not in the out-of-formulary purchases" in r.json()["detail"]
    w = get(client, "/spend/cash/overview", Plant=KR, Month=AUG, Segment="consumables")["with_oof"]
    assert w["oof"]["margin_loss"] == 0.0


def test_a_net_loss_below_zero_keeps_every_share_between_0_and_100(client, data):
    # DR: 600 gained on r8, 2200 earned more on r9 -> net -1600; shares are of the 600 gained
    items = get(client, "/spend/oof/items", Plant=DR, Month=AUG)["data"]
    assert [(r["material"], r["margin_loss"], r["share_pct"]) for r in items] == \
        [("101100", 600.0, 100.0), ("201300", -2200.0, 0.0)]
    vendors = get(client, "/spend/oof/vendors", Plant=DR, Month=AUG)["data"]
    assert [(v["vendor"], v["share_pct"]) for v in vendors] == [(ALPHA, 100.0), (BETA, 0.0)]


# ---------- HCG's own figure ----------

def test_hcgs_figure_beside_ours(client, data):
    d = oof(client, Plant=KR, Month=AUG)
    assert d["headline"]["hcg_figure"] == 2820.0
    assert d["hcg"] == {"months": [AUG], "centres": 1, "actual_margin": 7000.0, "potential_margin": 9820.0,
                        "margin_loss": 2820.0, "loss_pct": pytest.approx(28.7169, abs=1e-4), "diff": 0.0,
                        "matches": True}
    assert any("Margin Value Difference" in n for n in d["notes"])
    # every hospital: HCG also lists HC13 (loss 50), which we do not have -> off by 50
    a = oof(client, Month=AUG)["hcg"]
    assert (a["margin_loss"], a["diff"], a["centres"], a["matches"]) == (1270.0, -50.0, 3, False)


def test_no_hcg_figure_under_a_segment_or_brand_or_a_month_hcg_lacks(client, data):
    for params in ({"Segment": "onco"}, {"Material": "101100"}):
        d = oof(client, Plant=KR, Month=AUG, **params)
        assert d["hcg"] is None and d["headline"]["hcg_figure"] is None, params
        assert all(t["hcg_loss"] is None for t in d["trend"]) and all(x["hcg_loss"] is None for x in d["by_hospital"])
        assert any("per hospital and month only" in n for n in d["notes"]), params
    # Jul has no HCG Summary: no figure beside a total that includes it, the Aug point stays
    d = oof(client, Plant=KR, Month="ALL")
    assert d["hcg"] is None and d["headline"]["hcg_figure"] is None
    assert [(t["period"], t["hcg_loss"]) for t in d["trend"]] == [(JUL, None), (AUG, 2820.0)]
    assert any("covers Aug-26 only" in n for n in d["notes"])


# ---------- breakdowns ----------

def test_trend_every_purchase_month_whatever_the_month_filter(client, data):
    t = oof(client, Plant=KR, Month=AUG)["trend"]
    assert [(x["label"], x["margin_loss"], x["pct"]) for x in t] == \
        [("Jul-26", 1800.0, pytest.approx(1800 / 4300 * 100, abs=1e-4)), ("Aug-26", 2820.0, pytest.approx(28.7169, abs=1e-4))]
    assert (t[1]["actual"], t[1]["potential"], t[1]["loss"]) == (7000.0, 9820.0, 2820.0)


def test_hospital_ranking_keeps_negative_losses_and_hcgs_centres(client, data):
    d = oof(client, Plant=DR, Month=AUG)
    assert [(x["plant"], x["margin_loss"], x["hcg_loss"], x["rank"], x["selected"]) for x in d["by_hospital"]] == \
        [(KR, 2820.0, 2820.0, 1, False), (HC13, 0.0, 50.0, 2, False), (DR, -1600.0, -1600.0, 3, True)]
    assert d["by_hospital"][0]["his_name"] == "HCG KR, Bangalore"
    assert d["network"] == {"loss_pct": pytest.approx(1220 / 12220 * 100, abs=1e-4)}
    # DR's own loss is negative: -1600 / possible 2400
    assert d["headline"]["margin_loss"] == -1600.0
    assert d["headline"]["loss_pct"] == pytest.approx(-1600 / 2400 * 100, abs=1e-4)


def test_segment_split_lists_the_material_types_bought(client, data):
    cats = oof(client, Plant=KR, Month=AUG, Segment="onco")["by_category"]       # ignores the Segment filter
    assert [(c["key"], c["margin_loss"], c["purchase_value"]) for c in cats] == \
        [("onco", 3320.0, 6800.0), ("non-onco", -500.0, 2300.0)]               # no consumables bought


def test_top_switches_current_brand_better_vendors_and_no_alternative(client, data):
    d = oof(client, Plant=KR, Month=AUG)
    assert [(t["material"], t["suggested_material"], t["suggested_by"], t["margin_loss"]) for t in d["top_items"]] == \
        [("101100", "101900", PO.HCG_SHEET, 3000.0), ("201200", "201900", PO.HCG_SHEET, 600.0),
         ("101600", "101901", PO.FORMULARY_RULE, 320.0), ("201300", "201901", PO.HCG_SHEET, -1100.0)]
    # share = the part of what switching gains (3000 + 600 + 320), not of the net 2820
    assert d["top_items"][0]["share_of_total_pct"] == pytest.approx(3000 / 3920 * 100, abs=1e-4)
    assert d["top_items"][3]["share_of_total_pct"] == 0.0
    rule = d["top_items"][2]
    assert rule["estimated"] is True and rule["rule"] == PO.RULE_LEVELS["A"] + " (molecule matched by spelling)"
    assert [(t["material"], t["margin_loss"]) for t in d["better_items"]] == [("201300", -1100.0)]
    assert [(v["vendor"], v["vendor_codes"], v["vendor_name"], v["margin_loss"], v["no_alternative_value"])
            for v in d["top_vendors"]] == [(ALPHA, [V1], "Alpha Pharma", 3000.0, 500.0),
                                           (BETA, [V2], "Beta Distributors", -180.0, 100.0)]
    assert [(n["material"], n["status"], n["purchase_value"]) for n in d["no_alternative"]] == \
        [("201400", PO.NO_ALTERNATIVE, 500.0), ("201700", PO.UNPRICED, 100.0)]


def test_an_empty_selection_is_zero_with_null_percentages(client, data):
    d = oof(client, Plant="HS01", Month=AUG)
    h = d["headline"]
    assert margins(h) == (0.0, 0.0, 0.0) and h["loss_pct"] is None and h["purchase_value"] == 0.0
    assert d["top_items"] == [] and d["no_alternative"] == []
    assert any("No out-of-formulary purchase at" in n for n in d["notes"])


# ---------- tables ----------

def test_items_one_row_per_brand_and_suggestion(client, data):
    d = get(client, "/spend/oof/items", Plant=KR, Month=AUG)
    assert d["total"] == 7
    assert [(r["material"], r["suggested_by"]) for r in d["data"]] == \
        [("101100", PO.HCG_SHEET), ("201200", PO.HCG_SHEET), ("101600", PO.FORMULARY_RULE),
         ("201300", PO.HCG_SHEET), ("201400", PO.NONE), ("201500", PO.PARKED), ("201700", PO.NONE)]
    a = d["data"][0]
    assert (a["material_desc"], a["minor_group"], a["manufacturer"], a["suggested_desc"]) == \
        ("ONCO OOF A 100MG INJ", "MOLECULE A", "MAKER ONE", "ONCO FORM A 100MG INJ")
    assert (a["base_price_min"], a["sug_price"], a["sug_mrp"], a["sug_gst"]) == (600.0, 300.0, 1050.0, 5.0)
    assert (a["used_margin"], a["suggested_margin"], a["margin_gap"]) == (400.0, 700.0, 300.0)      # per unit
    assert margins(a) == (4000.0, 7000.0, 3000.0) and a["suggested_by_label"] == "HCG's worksheet"
    parked = d["data"][5]
    assert (parked["status"], parked["used_margin"], parked["margin_loss"], parked["purchase_value"]) == \
        (PO.PARKED, None, 0.0, 300.0)
    assert d["sums"] == {"qty": 69.0, "purchase_value": 9100.0, "value_after_discount": 8190.0,
                         "counted_value": 8200.0, "actual_margin": 7000.0, "suggested_value": 5380.0,
                         "potential_margin": 9820.0, "margin_loss": 2820.0, "gain_loss": 3920.0,
                         "better_loss": -1100.0, "lines": 8}


def test_items_keep_each_suggestion_source_apart_across_months(client, data):
    rows = get(client, "/spend/oof/items", Plant=KR, Month="ALL")["data"]
    para = [(r["suggested_by"], r["margin_loss"]) for r in rows if r["material"] == "201200"]
    assert para == [(PO.HCG_SHEET, 600.0), (PO.HCG_LIST, 300.0)]
    onco = [r for r in rows if r["material"] == "101100"]
    assert len(onco) == 1 and onco[0]["margin_loss"] == 4500.0                                 # Aug 3000 + Jul 1500


def test_items_filters_vendor_and_brand(client, data):
    d = get(client, "/spend/oof/items", Plant=KR, Month=AUG, facets=1)
    by = {f["field"]: f for f in d["facets"]}
    assert {"suggested_by", "minor_group", "manufacturer", "category"} <= set(by)
    assert get(client, "/spend/oof/items", Plant=KR, Month=AUG, **{"f.suggested_by": PO.NONE})["sums"]["purchase_value"] == 600.0
    v = get(client, "/spend/oof/items", Plant=KR, Month=AUG, Vendor=V1, facets=1)
    assert [r["material"] for r in v["data"]] == ["101100", "201400", "201500"]
    assert {"param": "Vendor", "label": "Vendor", "value": V1, "value_label": "Alpha Pharma", "token": V1} in v["scope"]
    one = get(client, "/spend/oof/items", Month=AUG, Material="101100")
    assert [(r["material"], r["margin_loss"], r["hospitals"]) for r in one["data"]] == [("101100", 3600.0, 2)]


def test_vendors_most_missed_first(client, data):
    d = get(client, "/spend/oof/vendors", Plant=KR, Month=AUG)
    assert d["total"] == 2
    a, b = d["data"]
    assert (a["vendor"], a["vendor_codes"], a["vendor_name"], a["brands"], a["purchase_value"], a["margin_loss"]) == \
        (ALPHA, [V1], "Alpha Pharma", 3, 6800.0, 3000.0)
    assert (b["vendor"], b["margin_loss"], b["gain_loss"], b["better_loss"], b["no_alternative_value"]) == \
        (BETA, -180.0, 920.0, -1100.0, 100.0)
    assert (a["share_pct"], b["share_pct"]) == (pytest.approx(3000 / 3920 * 100, abs=1e-4),
                                                pytest.approx(920 / 3920 * 100, abs=1e-4))
    assert d["sums"]["margin_loss"] == 2820.0 and d["sums"]["no_alternative_value"] == 600.0
    one = get(client, "/spend/oof/vendors", Month=AUG, Material="201300")["data"]
    assert [(r["vendor"], r["margin_loss"], r["hospitals"]) for r in one] == [(BETA, -3300.0, 2)]



def _second_code_for_alpha(data, name: str) -> str:
    """Re-key r4 (KR Aug, no alternative, 500 spent) to a second SAP code of Alpha Pharma, its
    name spelt with other spacing and case, as the GRN does for one supplier's regional codes."""
    v3 = "50003"
    p = data / "purchase" / f"{PC.OOF}.parquet"
    df = pd.read_parquet(p)
    r4 = df["material"].eq("201400")
    df.loc[r4, ["vendor_code", "vendor_name"]] = [v3, name]
    df.to_parquet(p, index=False)
    spend.clear_caches()
    return v3


def test_two_codes_of_one_supplier_are_one_vendor(client, data):
    v3 = _second_code_for_alpha(data, "  alpha   PHARMA ")
    d = get(client, "/spend/oof/vendors", Plant=KR, Month=AUG)
    assert d["total"] == 2
    a = d["data"][0]
    assert (a["vendor"], a["vendor_codes"], a["brands"], a["purchase_value"], a["no_alternative_value"]) == \
        (ALPHA, [V1, v3], 3, 6800.0, 500.0)
    o = oof(client, Plant=KR, Month=AUG)
    assert o["headline"]["vendors"] == 2
    assert [(v["vendor"], v["vendor_codes"]) for v in o["top_vendors"]] == [(ALPHA, [V1, v3]), (BETA, [V2])]


def test_vendor_filter_takes_the_name_or_one_code(client, data):
    v3 = _second_code_for_alpha(data, "ALPHA PHARMA")
    by_name = get(client, "/spend/oof/items", Plant=KR, Month=AUG, Vendor="alpha pharma")
    assert [r["material"] for r in by_name["data"]] == ["101100", "201400", "201500"]
    by_code = get(client, "/spend/oof/items", Plant=KR, Month=AUG, Vendor=v3)
    assert [r["material"] for r in by_code["data"]] == ["201400"]
    assert get(client, "/spend/oof/items", Plant=KR, Month=AUG, Vendor="NOBODY")["data"] == []

# ---------- the cash overview's CASH(TPA) & OUT OF FORMULARY line ----------

def test_with_oof_adds_the_purchases_of_the_same_hospitals_and_months(client, data):
    w = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)["with_oof"]
    # cash (tests/test_spend_api.py) 1680 / 4860 / 3180 + out of formulary 7000 / 9820 / 2820
    assert margins(w["cash"]) == pytest.approx((1680.0, 4860.0, 3180.0)) and margins(w["oof"]) == (7000.0, 9820.0, 2820.0)
    assert w["total"] == {"actual_margin": 8680.0, "potential_margin": 14680.0, "margin_loss": 6000.0,
                          "loss_pct": pytest.approx(6000 / 14680 * 100, abs=1e-4)}
    assert w["oof_months"] == [AUG] and w["coverage"][PO.NONE]["value"] == 600.0
    # Jul: cash c5 4 x 70 = 280; out of formulary 1500 + 300
    assert [(t["period"], t["cash_loss"], t["oof_loss"], t["loss"], t["has_oof"]) for t in w["trend"]] == \
        [(JUL, 280.0, 1800.0, 2080.0, True), (AUG, 3180.0, 2820.0, 6000.0, True)]
    assert "Payer and Type filters apply to the cash & TPA part only" in w["note"]


def test_with_oof_payer_and_type_cut_the_cash_part_only(client, data):
    w = get(client, "/spend/cash/overview", Plant=KR, Month=AUG, Payer="CASH(TPA)", Type="OP")["with_oof"]
    assert margins(w["cash"]) == pytest.approx((517.5, 1552.5, 1035.0))                       # c2 only
    assert margins(w["oof"]) == (7000.0, 9820.0, 2820.0)
    assert w["total"]["margin_loss"] == pytest.approx(3855.0)
    seg = get(client, "/spend/cash/overview", Plant=KR, Month=AUG, Segment="onco")["with_oof"]
    assert seg["oof"]["margin_loss"] == 3320.0                                                # rows 1 + 6


# ---------- the Brand filter, meta ----------

def test_brand_filter_offers_every_out_of_formulary_brand_bought(client, data):
    d = get(client, "/spend/brands", Plant=KR, Month=AUG, Section="oof")
    got = [(b["material"], b["oof"]["missed"], b["oof"]["suggested_material"]) for b in d["brands"]]
    # the biggest gain first; no alternative, parked and unpriced brands miss nothing and have no suggestion
    assert got == [("101100", 3000.0, "101900"), ("201200", 600.0, "201900"), ("101600", 320.0, "101901"),
                   ("201400", 0.0, None), ("201500", 0.0, None), ("201700", 0.0, None), ("201300", -1100.0, "201901")]
    top = d["brands"][0]
    assert top["oof"]["suggested_desc"] == "ONCO FORM A 100MG INJ" and top["oof"]["pct"] == pytest.approx(3000 / 7000 * 100, abs=1e-4)
    assert d["brands"][3]["oof"]["suggested_desc"] is None and d["brands"][3]["oof"]["pct"] is None
    # the payer does not cut purchases; one brand by code
    tpa = get(client, "/spend/brands", Plant=KR, Month=AUG, Payer="CASH(TPA)", Material="101100")["brands"][0]
    assert tpa["oof"]["missed"] == 3000.0 and tpa["material_desc"] == "ONCO OOF A 100MG INJ"


def test_meta_names_the_purchase_months(client, data):
    m = get(client, "/spend/meta")
    assert (m["oof_built"], m["oof_months"], m["oof_latest"], m["oof_built_at"]) == (True, [JUL, AUG], AUG, BUILT_AT)
    assert [s["key"] for s in m["oof_suggested_by"] if s["estimated"]] == [PO.FORMULARY_RULE]
    assert "python -m app.etl.purchase" in m["sources"]["out_of_formulary"]
    assert oof(client, Month=AUG)["data_as_of"] == BUILT_AT


# ---------- data handling ----------

def test_out_of_formulary_needs_no_spend_build(client, data, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SPEND_DATA_DIR", str(tmp_path / "spend-not-built"))
    spend.clear_caches()
    d = oof(client, Plant=KR, Month=AUG)
    assert d["headline"]["margin_loss"] == 2820.0 and d["scope"]["his_name"] == "HCG KR, Bangalore"
    assert d["by_hospital"][0]["his_code"] == "KABHK"                          # from the hospital map
    assert client.get("/spend/cash/overview").status_code == 503


def test_a_rebuild_is_served_without_a_restart(client, data):
    assert oof(client, Plant=KR, Month=AUG)["headline"]["margin_loss"] == 2820.0
    p = data / "purchase" / f"{PC.OOF}.parquet"
    df = pd.read_parquet(p)
    first = (df["material"] == "101100") & (df["plant"] == KR) & (df["period"] == AUG)
    df.loc[first, "margin_loss"] += 100.0
    df.to_parquet(p, index=False)
    assert oof(client, Plant=KR, Month=AUG)["headline"]["margin_loss"] == 2920.0


def test_no_purchase_identifier_in_any_response(client, data):
    bad = ("gr_no", "grn_no", "po_no", "po_number", "invoice", "reference_no", "uhid", "patname", "doctor_id")
    for path in ("/spend/oof/overview", "/spend/oof/items", "/spend/oof/vendors"):
        text = json.dumps(get(client, path, Month="ALL", facets=1)).lower()
        assert not any(f'"{b}' in text for b in bad), path


# ---------- HCG's real figures, on the committed purchase tables ----------

_REAL = Path(settings.PURCHASE_DATA_DIR)
_real = pytest.mark.skipif(not (_REAL / f"{PC.OOF}.parquet").exists() or
                           not (_REAL / f"{PC.HCG_SUMMARY}.parquet").exists(),
                           reason="purchase tables not built — run python -m app.etl.purchase")
# HCG's Summary 'Margin Value Difference' (rupees and paise): KR, and every centre, Aug-26;
# HCG's spend email rounds KR to 13,23,583. The cash & TPA loss is the email's KR Aug-26 line.
KR_AUG_LOSS, ALL_AUG_LOSS, KR_AUG_CASH_LOSS = 1_323_582.58, 3_243_199.13, 7_808_324.11


@pytest.fixture()
def real():
    spend.clear_caches()
    yield pd.read_parquet(_REAL / f"{PC.HCG_SUMMARY}.parquet")
    spend.clear_caches()


@_real
def test_real_kr_and_every_centre_equal_hcgs_summary(client, real):
    if AUG not in set(real["period"]):
        pytest.skip(f"HCG's Summary has no {AUG}")
    aug = real[real["period"] == AUG]
    kr = oof(client, Plant=KR, Month=AUG)
    assert kr["headline"]["margin_loss"] == KR_AUG_LOSS == round(float(aug.loc[aug["plant"] == KR, "loss"].sum()), 2)
    assert kr["headline"]["hcg_figure"] == KR_AUG_LOSS and kr["hcg"]["matches"]
    assert kr["headline"]["loss_pct"] == pytest.approx(72.87, abs=0.005)
    every = oof(client, Month=AUG)
    assert every["headline"]["margin_loss"] == ALL_AUG_LOSS == round(float(aug["loss"].sum()), 2)
    assert every["hcg"]["diff"] == 0.0
    # each centre, each month: ours equals HCG's to the paisa (each side rounded on its own)
    for x in every["by_hospital"]:
        assert x["margin_loss"] == pytest.approx(x["hcg_loss"], abs=0.011), x["plant"]
    for t in oof(client, Month="ALL")["trend"]:
        assert t["margin_loss"] == pytest.approx(t["hcg_loss"], abs=0.011), t["period"]


@_real
def test_real_kr_cash_and_out_of_formulary_line(client, real):
    w = get(client, "/spend/cash/overview", Plant=KR, Month=AUG)["with_oof"]
    if w is None or not w["cash"]["lines"]:
        pytest.skip("spend tables not built")
    assert (w["cash"]["margin_loss"], w["oof"]["margin_loss"]) == (KR_AUG_CASH_LOSS, KR_AUG_LOSS)
    assert w["total"]["margin_loss"] == pytest.approx(KR_AUG_CASH_LOSS + KR_AUG_LOSS, abs=0.005)
