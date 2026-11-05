# tests/test_purchase_insights_api.py — /spend/prices/* and /spend/generic/overview.
#
# The two published tables are written here from 11 invented price rows (two products, two
# hospitals, July and August, one supplier holding two SAP codes, a line without a product key,
# free stock and a Rs 0.01 placeholder) and 8 invented generic vs branded rows (an originator
# brand, its generic, an override, a branded brand cheaper than its generic, a month without an
# Item Master). Every expected number below was worked out by hand from them (comments show the
# arithmetic). A second, separate set of 10 price rows (CHECK_ROWS) gives each flag a product's
# purchases carry a case of its own: the unusually low list, the dots, the purchases that set the
# usual cheapest. The last tests read the COMMITTED tables (skipped until python -m
# app.etl.purchase has published) and set the API beside the build's own report and the
# figures read when the definitions were chosen.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.api import purchase_insights as PI
from app.core.config import settings
from app.etl.purchase import config as C
from app.etl.purchase import generic as Gn
from app.etl.purchase import prices as P

from tests.test_formulary_api import write_hospital_map

MAY, JUL, AUG = "2026-05", "2026-07", "2026-08"
KR, DR = "HC05", "HC06"
CETUX = ("CETUXIMAB|100MG|INJ|", "CETUXIMAB", "100MG", "INJ")
PARA = ("PARACETAMOL|500MG|TAB|", "PARACETAMOL", "500MG", "TAB")
PANI = ("PANITUMUMAB|100MG|INJ|", "PANITUMUMAB", "100MG", "INJ")
NO_KEY = (None, None, None, None)
VARDHMAN, SDH = "Vardhman Health Specialities", "Speciality Drug House"
# SAP supplier code -> name: Vardhman holds two codes, as in HCG's GRN (one per region)
VENDORS = {"50790": VARDHMAN, "50990": VARDHMAN, "52366": SDH}

# material -> (description, minor group, manufacturer, product)
BRANDS = {
    "101001": ("ERBITUX 100MG INJ VIAL", "CETUXIMAB", "MERCK", CETUX),
    "101002": ("CETUXA 100MG INJ VIAL", "CETUXIMAB", "INTAS", CETUX),
    "201001": ("PARA A 500MG TAB", "PARACETAMOL", "MICRO LABS", PARA),
    "201002": ("PARA B 500MG TAB", "PARACETAMOL", "CIPLA", PARA),
    "201003": ("SYRUP X", "MULTIVITAMIN", "ABBOTT", NO_KEY),
}
# period, plant, material, supplier code, price status, qty, value (one GRN line each)
PRICE_ROWS = [
    (AUG, KR, "101001", "50790", "ok", 10, 180000.0),    # p1  18,000 a vial
    (AUG, DR, "101001", "50990", "ok", 5, 85000.0),      # p2  17,000
    (AUG, KR, "101002", "52366", "ok", 4, 20000.0),      # p3   5,000  <- August's cheapest cetuximab
    (AUG, DR, "101002", "50990", "ok", 2, 12000.0),      # p4   6,000
    (AUG, KR, "201001", "50790", "ok", 100, 200.0),      # p5   2.00
    (AUG, DR, "201002", "52366", "ok", 100, 150.0),      # p6   1.50   <- the cheapest paracetamol
    (AUG, KR, "201003", "50790", "ok", 1, 999.0),        # p7   no product key: not compared
    (AUG, KR, "101001", "50790", "no_price", 2, 0.0),    # p8   free vials
    (AUG, DR, "101002", "52366", "token", 1, 0.01),      # p9   a Rs 0.01 placeholder
    (JUL, KR, "101001", "50790", "ok", 3, 60000.0),      # p10 20,000
    (JUL, KR, "101002", "52366", "ok", 1, 4000.0),       # p11  4,000  <- July's cheapest
]
# Aug, every hospital, compared rows p1-p6 (each product's usual cheapest: the lowest price at
# which at least 10% of its quantity was bought — here the plain minimum, which holds 4 of 21
# cetuximab vials and 100 of 200 paracetamol tablets):
#   cetuximab: 297,000 for 21 vials; at 5,000 = 1,05,000 -> extra 1,92,000; dearest 18,000 (3.6x)
#   paracetamol: 350 for 200; at 1.50 = 300 -> extra 50; dearest 2.00
# -> extra 1,92,050 of 2,97,350 compared; value of every row 2,98,349.01

# material -> (description, minor group, manufacturer, class, class source, originator company, product)
GENERIC_BRANDS = {
    "101001": ("ERBITUX 100MG INJ VIAL", "CETUXIMAB", "MERCK", "branded", "originator_list", "Merck", CETUX),
    "101002": ("CETUXA 100MG INJ VIAL", "CETUXIMAB", "INTAS", "generic", "default", None, CETUX),
    "201001": ("PARA A 500MG TAB", "PARACETAMOL", "MICRO LABS", "generic", "default", None, PARA),
    "201005": ("CALPOL 500MG TAB", "PARACETAMOL", "GSK", "branded", "originator_list", "GSK", PARA),
    "100714": ("VECTIBIX 100MG INJ VIAL", "PANITUMUMAB", "DR REDDYS", "branded", "override", None, PANI),
}
# period, plant, material, Item Master flag, tier, qty, value, cheapest generic (material, price) that month
GENERIC_ROWS = [
    (AUG, KR, "101001", "FORMULARY", "R", 10, 180000.0, ("101002", 5000.0)),        # g1 (18,000 - 5,000) x 10 = 1,30,000
    (AUG, DR, "101001", "FORMULARY", "R", 5, 85000.0, ("101002", 5000.0)),          # g2 (17,000 - 5,000) x 5 = 60,000
    (AUG, KR, "101002", "OUT OF FORMULARY", None, 4, 20000.0, ("101002", 5000.0)),  # g3 the generic itself
    (AUG, KR, "201001", "FORMULARY", "P", 100, 200.0, ("201001", 2.0)),             # g4
    (AUG, DR, "100714", "NON FORMULARY", None, 2, 50000.0, None),                   # g5 override, no generic bought
    (AUG, DR, "201005", "FORMULARY", "P-SCH", 100, 150.0, ("201001", 2.0)),         # g6 branded, cheaper: extra 0
    (MAY, KR, "101001", None, None, 3, 60000.0, ("101002", 4000.0)),                # g7 no Item Master; 48,000
    (MAY, KR, "101002", None, None, 1, 4000.0, ("101002", 4000.0)),                 # g8
]
# Aug, every hospital: value 3,35,350 = branded 3,15,150 (g1, g2, g5, g6) + generic 20,200;
# extra vs generic 1,90,000 on 2,65,150 compared (g1, g2, g6); g5's 50,000 has no generic

DEFINITION = "Branded = an originator company's product (test definition)."


def _category(material: str) -> tuple:
    return ("Onco Drugs", "onco") if material.startswith("1") else ("Other Drugs", "non-onco")


def write_prices(folder: Path, price_rows: list, brands: dict, vendors: dict) -> None:
    """purchase_prices from (period, plant, material, supplier code, status, qty, value) rows,
    one GRN line each."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for period, plant, mat, code, status, qty, value in price_rows:
        desc, group, maker, (key, molecule, strength, form) = brands[mat]
        cat, seg = _category(mat)
        price = np.nan if status == P.NO_PRICE else value / qty
        rows.append({"period": period, "plant": plant, "category": cat, "segment": seg, "material": mat,
                     "material_desc": desc, "minor_group": group, "manufacturer": maker, "vendor_code": code,
                     "vendor_name": vendors[code], "price_status": status, "molecule_key": key,
                     "molecule": molecule, "strength": strength, "form": form, "kind": None, "lines": 1,
                     "qty": float(qty), "value": value, "unit_price_min": price, "unit_price_max": price})
    pd.DataFrame(rows)[P.PRICES_COLUMNS].to_parquet(folder / f"{C.PRICES}.parquet", index=False)


def write_tables(folder: Path) -> None:
    write_prices(folder, PRICE_ROWS, BRANDS, VENDORS)

    rows = []
    for period, plant, mat, flag, tier, qty, value, generic in GENERIC_ROWS:
        desc, group, maker, cls, source, company, (key, molecule, strength, form) = GENERIC_BRANDS[mat]
        cat, seg = _category(mat)
        compared = cls == Gn.BRANDED and generic is not None
        gp = generic[1] if generic else np.nan
        formulary = flag == "FORMULARY"
        rows.append({"period": period, "plant": plant, "category": cat, "segment": seg, "material": mat,
                     "material_desc": desc, "minor_group": group, "manufacturer": maker, "class": cls,
                     "class_source": source, "originator_company": company, "molecule_key": key,
                     "molecule": molecule, "strength": strength, "form": form, "kind": None,
                     "formulary_flag": flag, "formulary_class": (f"FORMULARY{tier}" if formulary else flag),
                     "tier": tier, "tier_group": Gn.TIER_GROUPS.get(tier) if formulary else None,
                     "generic_material": generic[0] if generic else None,
                     "generic_desc": GENERIC_BRANDS[generic[0]][0] if generic else None, "generic_price": gp,
                     "lines": 1, "qty": float(qty), "value": value,
                     "compared_value": value if compared else 0.0,
                     "extra_vs_generic": max(0.0, value / qty - gp) * qty if compared else 0.0})
    pd.DataFrame(rows)[Gn.GENERIC_COLUMNS].to_parquet(folder / f"{C.GENERIC}.parquet", index=False)

    (folder / C.VALIDATION_FILE).write_text(json.dumps({
        "built_at": "2026-10-22T12:00:00+00:00", "latest": AUG, "scope": {"file": C.SCOPE_FILE},
        "months": {AUG: {"funnel": [{"kind": "lines", "label": "lines in the GRN"},
                                    {"kind": "mvt_keep", "label": "movement type 101"},
                                    {"kind": "plant", "label": "not plants HM03, HM04"}]}},
        "generic": {"definition": DEFINITION, "estimated": True, "originator_patterns": 57,
                    "originator_companies": 43, "files": {"originators": Gn.ORIGINATOR_FILE,
                                                          "overrides": Gn.OVERRIDES_FILE},
                    "overrides": [{"material": "100714", "class": "branded", "reason": "licensed brand"}]}}))


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_tables(tmp_path / "purchase")
    write_hospital_map(tmp_path / "reference")
    PI.clear_caches()
    yield tmp_path
    PI.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def prices(client, **params):
    return get(client, "/spend/prices/overview", **params)


def generic(client, **params):
    return get(client, "/spend/generic/overview", **params)


def by_key(rows: list, key: str = "molecule_key") -> dict:
    return {r[key]: r for r in rows}


# ---------- price variation: headline and products ----------

def test_price_headline_by_hand(client, data):
    h = prices(client, Month=AUG)["headline"]
    assert (h["molecules"], h["molecules_with_spread"]) == (2, 2)
    assert (h["extra_paid_vs_cheapest"], h["compared_value"], h["value"]) == (192050.0, 297350.0, 298349.01)
    assert h["extra_pct"] == pytest.approx(192050 / 297350 * 100, abs=1e-4)
    assert h["compared_pct"] == pytest.approx(297350 / 298349.01 * 100, abs=1e-4)
    # 9 lines, 5 brands, 2 suppliers (Vardhman's two codes are one), 2 hospitals
    assert (h["lines"], h["brands"], h["vendors"], h["hospitals"]) == (9, 5, 2, 2)
    assert h["extra_paid_vs_network_cheapest"] is None


def test_products_are_set_against_the_cheapest(client, data):
    d = prices(client, Month=AUG)
    assert [m["molecule_key"] for m in d["molecules"]] == [CETUX[0], PARA[0]]       # most extra first
    c = get(client, "/spend/prices/molecules", Month=AUG)["data"][0]
    assert (c["cheapest_price"], c["min_price"], c["max_price"], c["spread_ratio"]) == (5000.0, 5000.0, 18000.0, 3.6)
    assert (c["qty"], c["value"], c["extra_paid"]) == (21.0, 297000.0, 192000.0)     # 2,97,000 - 21 x 5,000
    assert c["wavg_price"] == pytest.approx(297000 / 21, abs=1e-4)
    assert c["price_index"] == pytest.approx(297000 / 105000, abs=1e-4)
    assert c["extra_pct"] == pytest.approx(192000 / 297000 * 100, abs=1e-4)
    assert c["share_pct"] == pytest.approx(192000 / 192050 * 100, abs=1e-4)
    assert (c["brands"], c["vendors"], c["plants"], c["lines"], c["varies"]) == (2, 2, 2, 4, True)
    assert (c["cheapest_brand"], c["cheapest_vendor"], c["cheapest_plant"]) == ("CETUXA 100MG INJ VIAL", SDH, KR)
    assert (c["minor_group"], c["molecule"], c["strength"], c["form"], c["kind"], c["category"]) == \
        ("CETUXIMAB", "CETUXIMAB", "100MG", "INJ", None, "Onco Drugs")
    p = by_key(get(client, "/spend/prices/molecules", Month=AUG)["data"])[PARA[0]]
    assert (p["cheapest_price"], p["max_price"], p["wavg_price"], p["extra_paid"]) == (1.5, 2.0, 1.75, 50.0)
    assert p["price_index"] == pytest.approx(350 / 300, abs=1e-4)


def test_one_hospital_is_set_against_its_own_cheapest_and_the_networks(client, data):
    d = prices(client, Month=AUG, Plant=KR)
    h = d["headline"]
    # KR: cetuximab p1 + p3 = 2,00,000 for 14 at KR's own 5,000 -> 1,30,000; paracetamol only p5 -> 0
    assert (h["extra_paid_vs_cheapest"], h["molecules"], h["molecules_with_spread"]) == (130000.0, 2, 1)
    # against the cheapest of any hospital, paracetamol adds 200 - 100 x 1.50 = 50
    assert h["extra_paid_vs_network_cheapest"] == 130050.0
    assert (h["value"], h["lines"], h["hospitals"]) == (201199.0, 5, 1)
    assert d["scope"]["plant"] == KR and [x["selected"] for x in d["by_hospital"]] == [True, False]


def test_hospital_ranking_adds_up_to_the_network(client, data):
    hb = prices(client, Month=AUG, Plant=DR)["by_hospital"]       # the ranking ignores the hospital
    assert [(x["plant"], x["rank"], x["extra_paid"], x["at_cheapest"]) for x in hb] == \
        [(KR, 1, 130050.0, 70150.0), (DR, 2, 62000.0, 35150.0)]
    assert sum(x["extra_paid"] for x in hb) == prices(client, Month=AUG)["headline"]["extra_paid_vs_cheapest"]
    assert hb[1]["price_index"] == pytest.approx(97150 / 35150, abs=1e-4) and hb[1]["selected"]


def test_several_months_take_the_usual_cheapest_of_all(client, data):
    d = prices(client, Month="ALL")
    # cetuximab, both months: July's single vial at 4,000 is 1 of 25 (under 10%), so the usual
    # cheapest is August's 5,000 (cumulative 5 vials): above it p10 3 x 15,000 + p1 1,30,000 +
    # p2 60,000 + p4 2,000 = 2,37,000; p11 (under it) adds nothing; paracetamol 50
    assert d["headline"]["extra_paid_vs_cheapest"] == 237050.0
    assert any("usual cheapest over all of them" in n for n in d["notes"])
    # every month on its own: July 64,000 - 4 x 4,000 = 48,000; August 1,92,050
    assert [(m["label"], m["extra_paid"], m["compared_value"], m["molecules"], m["molecules_with_spread"],
             m["lines"]) for m in d["by_month"]] == [("Jul-26", 48000.0, 64000.0, 1, 1, 2),
                                                     ("Aug-26", 192050.0, 297350.0, 2, 2, 9)]
    assert prices(client, Month=AUG)["by_month"] == d["by_month"]                  # Month aside
    assert [m["period"] for m in d["months_available"]] == [JUL, AUG]


def test_one_odd_purchase_is_listed_not_used_as_the_cheapest(client, data):
    # one CETUXA vial at 1,000 (DR, Speciality Drug House): 1 of 22 vials, under 10% of the
    # quantity, so the usual cheapest stays 5,000 and nothing else moves; under half of it, the
    # purchase is listed as unusually low for HCG to check
    odd = (AUG, DR, "101002", "52366", "ok", 1, 1000.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(globals(), "PRICE_ROWS", PRICE_ROWS + [odd])
        write_tables(data / "purchase")
    PI.clear_caches()
    d = prices(client, Month=AUG, Molecule=CETUX[0])
    h = d["headline"]
    assert (h["extra_paid_vs_cheapest"], h["low_rows"], h["low_value"]) == (192050.0, 1, 1000.0)
    c = by_key(d["molecules"])[CETUX[0]]
    # low_rows: hospital x brand x supplier groups with a low row (one here: DR CETUXA SDH)
    assert (c["cheapest_price"], c["min_price"], c["low_rows"], c["extra_paid"]) == (5000.0, 1000.0, 1, 192000.0)
    low = d["low_prices"]
    assert (low["rows"], low["products"], len(low["items"])) == (1, 1, 1)
    i = low["items"][0]
    assert (i["plant"], i["material"], i["vendor_name"], i["unit_price"], i["cheapest_price"], i["below_cheapest"]) == \
        (DR, "101002", SDH, 1000.0, 5000.0, 4000.0)
    flagged = [(p["plant"], p["vendor_name"], p["unit_price"]) for p in d["molecule"]["points"] if p["unusually_low"]]
    assert flagged == [(DR, SDH, 1000.0)]
    assert any("unusually low" in n for n in d["notes"])


def test_vendors_are_suppliers_by_name(client, data):
    d = prices(client, Month=AUG)
    v, s = d["vendors"]
    # Vardhman (both codes): p1, p2, p4, p5 = 2,77,200; at the cheapest 50,000 + 25,000 + 10,000 + 150
    assert (v["vendor_name"], v["vendor_codes"], v["value"], v["at_cheapest"], v["extra_paid"]) == \
        (VARDHMAN, ["50790", "50990"], 277200.0, 85150.0, 192050.0)
    assert v["price_index"] == pytest.approx(277200 / 85150, abs=1e-4)
    assert (v["molecules"], v["brands"], v["plants"], v["lines"]) == (2, 3, 2, 4)
    assert (s["vendor_name"], s["value"], s["extra_paid"], s["price_index"]) == (SDH, 20150.0, 0.0, 1.0)
    assert v["share_pct"] + s["share_pct"] == pytest.approx(100.0) and d["vendors_total"] == 2


def test_the_chosen_product_shows_its_purchases_and_vendors(client, data):
    m = prices(client, Month=AUG)["molecule"]                 # default: the most extra paid
    assert (m["molecule_key"], m["extra_paid"]) == (CETUX[0], 192000.0)
    assert [(p["plant"], p["material"], p["vendor_name"], p["unit_price"], p["qty"], p["value"]) for p in m["points"]] == [
        (KR, "101002", SDH, 5000.0, 4.0, 20000.0), (DR, "101002", VARDHMAN, 6000.0, 2.0, 12000.0),
        (DR, "101001", VARDHMAN, 17000.0, 5.0, 85000.0), (KR, "101001", VARDHMAN, 18000.0, 10.0, 180000.0)]
    # p3 (KR CETUXA Speciality Drug House at 5,000) set August's cheapest: its dot and supplier are marked
    assert [p["sets_cheapest"] for p in m["points"]] == [True, False, False, False]
    assert not any(p["unusually_low"] for p in m["points"])
    assert m["vendors"] == 2                               # the count; by_vendor lists them
    v = by_key(m["by_vendor"], "vendor_name")
    assert (v[VARDHMAN]["value"], v[VARDHMAN]["extra_paid"], v[VARDHMAN]["qty"]) == (277000.0, 192000.0, 17.0)
    assert (v[VARDHMAN]["sets_cheapest"], v[SDH]["sets_cheapest"]) == (False, True)
    assert v[VARDHMAN]["wavg_price"] == pytest.approx(277000 / 17, abs=1e-4)
    assert (v[SDH]["wavg_price"], v[SDH]["price_index"]) == (5000.0, 1.0)
    d = prices(client, Month=AUG, Molecule="paracetamol|500mg|tab|")      # any case
    assert d["scope"]["molecule"] == PARA[0]
    assert [(p["material"], p["unit_price"]) for p in d["molecule"]["points"]] == [("201002", 1.5), ("201001", 2.0)]


def test_a_product_not_bought_is_a_note_not_an_error(client, data):
    d = prices(client, Month=JUL, Molecule=PARA[0])
    assert d["molecule"] is None and any("not bought" in n for n in d["notes"])


def test_vendor_and_brand_drills_keep_the_selections_cheapest(client, data):
    d = prices(client, Month=AUG, Vendor="speciality drug house")
    # p3 and p6 were the cheapest: nothing above it; p9's 0.01 is in the value only
    assert (d["headline"]["extra_paid_vs_cheapest"], d["headline"]["molecules_with_spread"]) == (0.0, 0)
    assert (d["headline"]["value"], d["headline"]["vendors"]) == (20150.01, 1)
    assert any("vendor or brand is selected" in n and "usual cheapest" in n for n in d["notes"])
    # one SAP code: p2 + p4 = 97,000 for 7 vials at 5,000 -> 62,000
    assert prices(client, Month=AUG, Vendor="50990.0")["headline"]["extra_paid_vs_cheapest"] == 62000.0
    # the supplier's name, spacing and case aside, takes both codes
    assert prices(client, Month=AUG, Vendor="  VARDHMAN  health specialities ")["headline"][
        "extra_paid_vs_cheapest"] == 192050.0
    # one brand: ERBITUX p1 + p2 = 2,65,000 for 15 at CETUXA's 5,000 -> 1,90,000
    c = get(client, "/spend/prices/molecules", Month=AUG, Material="101001")["data"]
    assert [(r["molecule_key"], r["min_price"], r["cheapest_price"], r["extra_paid"], r["brands"]) for r in c] == \
        [(CETUX[0], 17000.0, 5000.0, 190000.0, 1)]
    assert c[0]["price_index"] == pytest.approx(265000 / 75000, abs=1e-4) and c[0]["spread_ratio"] == 3.6


def test_what_was_not_compared(client, data):
    nc = {x["reason"]: (x["lines"], x["value"]) for x in prices(client, Month=AUG)["not_compared"]}
    assert nc == {"no_product": (1, 999.0), "no_price": (1, 0.0), "token": (1, 0.01), "other_unit": (0, 0.0)}


def test_segment_and_category_cut_the_products(client, data):
    h = prices(client, Month=AUG, Segment="non-onco")["headline"]
    assert (h["molecules"], h["extra_paid_vs_cheapest"], h["value"]) == (1, 50.0, 1349.0)   # p5, p6, p7
    h = prices(client, Month=AUG, Category="Onco Drugs")["headline"]
    assert (h["molecules"], h["extra_paid_vs_cheapest"], h["value"]) == (1, 192000.0, 297000.01)


def test_molecules_list_is_contract_a(client, data):
    d = get(client, "/spend/prices/molecules", Month=AUG)
    assert d["total"] == 2 and d["sums"] == {"value": 297350.0, "extra_paid": 192050.0, "lines": 6}
    d = get(client, "/spend/prices/molecules", Month=AUG, page=0, page_size=1, sort_field="extra_paid",
            sort_order="asc")
    assert d["total"] == 2 and d["data"][0]["molecule_key"] == PARA[0]
    assert [r["molecule_key"] for r in get(client, "/spend/prices/molecules", Month=AUG,
                                           global_filter="cetux")["data"]] == [CETUX[0]]
    d = get(client, "/spend/prices/molecules", Month=AUG, **{"f.form": "TAB"})
    assert d["total"] == 1 and d["sums"] == {"value": 350.0, "extra_paid": 50.0, "lines": 2}
    d = get(client, "/spend/prices/molecules", Month=AUG, Vendor=SDH, facets="only")
    assert {f["field"] for f in d["facets"]} >= {"category", "form", "varies"}
    # varies = the dearest at least 1% above the USUAL cheapest, so not 'bought at more than one price'
    assert next(f["label"] for f in d["facets"] if f["field"] == "varies") == "Dearest above the usual cheapest"
    assert [(c["param"], c["value_label"]) for c in d["scope"]] == \
        [("Plant", "All hospitals"), ("Month", "Aug-26"), ("Vendor", SDH)]


def test_limit_caps_the_chart(client, data):
    d = prices(client, Month=AUG, limit=1)
    assert len(d["molecules"]) == 1 and d["molecules_total"] == 2
    assert len(prices(client, Month=AUG, limit=0)["molecules"]) == 1


def test_notes_say_which_receipts_count_from_the_build_report(client, data):
    for path in ("/spend/prices/overview", "/spend/generic/overview"):
        notes = get(client, path)["notes"]
        assert notes[0] == ("Purchases: the goods receipts HCG's out-of-formulary workings count — movement type "
                            "101; not plants HM03, HM04 (purchase_scope_exclusions.csv); every month with a GRN "
                            "file, every formulary class."), path
    assert any(n.startswith("Not compared") and "placeholder" in n for n in prices(client)["notes"])


def test_type_and_payer_do_not_apply_and_say_so(client, data):
    for path in ("/spend/prices/overview", "/spend/generic/overview"):
        d = get(client, path, Month=AUG, Type="IP", Payer="CASH")
        assert d["scope"]["ignored"] == ["Type", "Payer"], path
        assert any("no patient type or payer" in n for n in d["notes"]), path
        assert "ignored" not in get(client, path, Month=AUG, Type="ALL")["scope"], path
    assert prices(client, Month=AUG, Payer="GOV")["headline"]["extra_paid_vs_cheapest"] == 192050.0


# ---------- price variation: the low list, a product's dots, the purchases that set the cheapest ----------
# A SEPARATE set of invented rows (PRICE_ROWS above is left as it is): one injection bought in July
# and August from three suppliers at both hospitals, and one tablet bought in August.
TRAS = ("TRASTUZUMAB|440MG|INJ|", "TRASTUZUMAB", "440MG", "INJ")
ONDA = ("ONDANSETRON|4MG|TAB|", "ONDANSETRON", "4MG", "TAB")
CITY = "City Pharma"
CHECK_VENDORS = {**VENDORS, "53000": CITY}
CHECK_BRANDS = {
    "105001": ("HERCLON 440MG INJ VIAL", "TRASTUZUMAB", "ROCHE", TRAS),
    "105002": ("VIVITRA 440MG INJ VIAL", "TRASTUZUMAB", "ZYDUS", TRAS),
    "205001": ("EMESET 4MG TAB", "ONDANSETRON", "CIPLA", ONDA),
}
CHECK_ROWS = [
    (JUL, KR, "105002", "52366", "ok", 1, 4000.0),       # c1   4,000  KR VIVITRA SDH in July
    (AUG, KR, "105002", "52366", "ok", 9, 108000.0),     # c2  12,000  ... the same group in August
    (AUG, DR, "105002", "50790", "ok", 1, 3000.0),       # c3   3,000  Vardhman's two SAP codes: one
    (AUG, DR, "105002", "50990", "ok", 1, 3000.0),       # c4   3,000  group, DR VIVITRA Vardhman
    (JUL, DR, "105002", "53000", "ok", 1, 8000.0),       # c5   8,000  City Pharma's only purchase
    (AUG, DR, "105001", "52366", "ok", 5, 50000.0),      # c6  10,000
    (AUG, KR, "105001", "50790", "ok", 20, 300000.0),    # c7  15,000
    (JUL, KR, "105001", "50790", "ok", 10, 140000.0),    # c8  14,000
    (AUG, KR, "205001", "50790", "ok", 100, 50000.0),    # c9     500
    (AUG, DR, "205001", "52366", "ok", 10, 2000.0),      # c10    200
]
BOTH = f"{JUL},{AUG}"
# July + August, every hospital:
#   trastuzumab, 48 vials, cheapest first: c3 1, c4 2, c1 3, c5 4, c6 9 >= 4.8 (10%) -> usual
#   cheapest 10,000, set by c6 (DR HERCLON Speciality Drug House); unusually low = under 5,000:
#   c1, c3, c4. c5 (8,000) is under the cheapest but not under half of it.
#   ondansetron, 110 tablets: c10 10 < 11, then c9 -> 500, set by c9; under 250: c10.
# Extra paid: c2 1,08,000 - 90,000 = 18,000; c7 3,00,000 - 2,00,000 = 1,00,000; c8 1,40,000 -
#   1,00,000 = 40,000 -> 1,58,000. c1, c3, c4, c5 and c10 are under their cheapest and add 0
#   (netted off they would take 6,000 + 7,000 + 7,000 + 2,000 + 3,000 away: 1,33,000).
# Low groups, hospital x brand x supplier, each with its low rows only: DR VIVITRA Vardhman (c3 + c4:
#   2 vials, 6,000; 20,000 - 6,000 = 14,000 under), KR VIVITRA SDH (c1: 4,000; 6,000 under), DR
#   EMESET SDH (c10: 2,000; 5,000 - 2,000 = 3,000 under) -> 3 groups, Rs 12,000. Trastuzumab has 2
#   of them, from 3 table rows; KR VIVITRA SDH's two months average 1,12,000 / 10 = 11,200, far
#   above half the cheapest, yet its July vial is low.


@pytest.fixture()
def checks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_prices(tmp_path / "purchase", CHECK_ROWS, CHECK_BRANDS, CHECK_VENDORS)   # the prices table only
    write_hospital_map(tmp_path / "reference")
    PI.clear_caches()
    yield tmp_path
    PI.clear_caches()


def test_the_low_list_counts_hospital_brand_supplier_groups(client, checks):
    d = prices(client, Month=BOTH)
    low = d["low_prices"]
    assert (low["rows"], low["value"], low["products"]) == (3, 12000.0, 2)
    assert [(i["molecule_key"], i["plant"], i["material"], i["vendor_name"], i["qty"], i["value"], i["unit_price"],
             i["cheapest_price"], i["below_cheapest"], i["lines"]) for i in low["items"]] == [
        (TRAS[0], DR, "105002", VARDHMAN, 2.0, 6000.0, 3000.0, 10000.0, 14000.0, 2),
        (TRAS[0], KR, "105002", SDH, 1.0, 4000.0, 4000.0, 10000.0, 6000.0, 1),
        (ONDA[0], DR, "205001", SDH, 10.0, 2000.0, 200.0, 500.0, 3000.0, 1)]
    assert (d["headline"]["low_rows"], d["headline"]["low_value"]) == (3, 12000.0)
    t, o = by_key(d["molecules"])[TRAS[0]], by_key(d["molecules"])[ONDA[0]]
    assert (t["low_rows"], o["low_rows"]) == (2, 1)                         # groups, not table rows (3 + 1)
    listed = by_key(get(client, "/spend/prices/molecules", Month=BOTH)["data"])
    assert (listed[TRAS[0]]["low_value"], listed[ONDA[0]]["low_value"]) == (10000.0, 2000.0)


@pytest.mark.parametrize("params, groups", [
    ({"Month": BOTH}, 3),
    ({"Month": AUG}, 2),            # August's own cheapest is 10,000 too (c3 1, c4 2, c6 7 >= 3.6): c3 + c4, c10
    ({"Month": JUL}, 0),            # July's cheapest is 8,000 (c1 1 < 1.2, c5 2): c1 is exactly half, not under it
    ({"Month": "ALL", "Plant": KR}, 1),        # KR's own cheapest 12,000 (c1 1 < 4, c2 10): c1 under 6,000
    ({"Month": "ALL", "Vendor": VARDHMAN}, 1),  # set against everyone's 10,000: c3 + c4
    ({"Month": "ALL", "Segment": "onco"}, 2),   # trastuzumab only
])
def test_the_products_low_rows_add_up_to_the_headline(client, checks, params, groups):
    d = prices(client, **params)
    listed = get(client, "/spend/prices/molecules", **params)["data"]
    assert d["headline"]["low_rows"] == d["low_prices"]["rows"] == groups, params
    assert sum(m["low_rows"] for m in d["molecules"]) == sum(m["low_rows"] for m in listed) == groups, params
    assert sum(m["low_value"] for m in listed) == pytest.approx(d["headline"]["low_value"]), params


def test_a_products_dots_flag_the_groups_the_low_list_lists(client, checks):
    d = prices(client, Month=BOTH, Molecule=TRAS[0])
    pts = d["molecule"]["points"]
    assert [(p["plant"], p["material"], p["vendor_name"], p["qty"], p["unit_price"], p["unusually_low"],
             p["sets_cheapest"]) for p in pts] == [
        (DR, "105002", VARDHMAN, 2.0, 3000.0, True, False),
        (DR, "105002", CITY, 1.0, 8000.0, False, False),          # under the cheapest, not under half of it
        (DR, "105001", SDH, 5.0, 10000.0, False, True),           # c6 set the usual cheapest
        (KR, "105002", SDH, 10.0, 11200.0, True, False),          # its average is not low; July's vial was
        (KR, "105001", VARDHMAN, 30.0, 14666.6667, False, False)]
    listed = {(i["plant"], i["material"], i["vendor_name"]) for i in d["low_prices"]["items"]
              if i["molecule_key"] == TRAS[0]}
    assert {(p["plant"], p["material"], p["vendor_name"]) for p in pts if p["unusually_low"]} == listed
    assert d["molecule"]["low_rows"] == len(listed) == 2


def test_the_purchases_that_set_the_usual_cheapest_are_marked(client, checks):
    m = prices(client, Month=BOTH, Molecule=TRAS[0])["molecule"]
    assert (m["cheapest_price"], m["cheapest_brand"], m["cheapest_vendor"], m["cheapest_plant"]) == \
        (10000.0, "HERCLON 440MG INJ VIAL", SDH, DR)
    assert [(v["vendor_name"], v["value"], v["at_cheapest"], v["extra_paid"], v["sets_cheapest"]) for v in m["by_vendor"]] == [
        (VARDHMAN, 446000.0, 320000.0, 140000.0, False),          # c3, c4, c7, c8
        (SDH, 162000.0, 150000.0, 18000.0, True),                 # c1, c2, c6
        (CITY, 8000.0, 10000.0, 0.0, False)]                      # c5
    # July alone: its own cheapest is 8,000, set by c5 — the mark moves with the selection
    m = prices(client, Month=JUL, Molecule=TRAS[0])["molecule"]
    assert m["cheapest_price"] == 8000.0
    assert [(p["vendor_name"], p["sets_cheapest"]) for p in m["points"] if p["sets_cheapest"]] == [(CITY, True)]
    assert [v["vendor_name"] for v in m["by_vendor"] if v["sets_cheapest"]] == [CITY]
    assert not any(p["unusually_low"] for p in m["points"])            # c1 at 4,000 is exactly half
    # KR alone: KR's cheapest is 12,000 (c2), and the same dot holds c1, under half of it
    kr = by_key(prices(client, Month="ALL", Plant=KR, Molecule=TRAS[0])["molecule"]["points"], "material")["105002"]
    assert (kr["unit_price"], kr["unusually_low"], kr["sets_cheapest"]) == (11200.0, True, True)
    # a supplier that never paid the cheapest: none of its dots is marked
    m = prices(client, Month=BOTH, Vendor=VARDHMAN, Molecule=TRAS[0])["molecule"]
    assert m["cheapest_price"] == 10000.0 and not any(p["sets_cheapest"] for p in m["points"])
    assert not any(v["sets_cheapest"] for v in m["by_vendor"])


def test_purchases_under_the_usual_cheapest_add_nothing(client, checks):
    d = prices(client, Month=BOTH)
    h = d["headline"]
    assert (h["extra_paid_vs_cheapest"], h["compared_value"], h["value"]) == (158000.0, 668000.0, 668000.0)
    assert [(m["molecule_key"], m["extra_paid"], m["varies"]) for m in d["molecules"]] == \
        [(TRAS[0], 158000.0, True), (ONDA[0], 0.0, False)]            # one low tablet, the rest at one price
    # a supplier or hospital that bought under the usual cheapest: nothing extra, a price index under 1
    city = by_key(prices(client, Month=BOTH, Molecule=TRAS[0])["molecule"]["by_vendor"], "vendor_name")[CITY]
    assert (city["extra_paid"], city["price_index"]) == (0.0, 0.8)
    assert prices(client, Month=BOTH, Vendor=CITY)["headline"]["extra_paid_vs_cheapest"] == 0.0
    hb = by_key(d["by_hospital"], "plant")
    # DR: 66,000 paid for what costs 85,000 at the cheapest (c3, c4, c5 10,000 each, c6 50,000, c10 5,000)
    assert (hb[DR]["extra_paid"], hb[DR]["value"], hb[DR]["at_cheapest"]) == (0.0, 66000.0, 85000.0)
    assert hb[DR]["price_index"] == pytest.approx(66000 / 85000, abs=1e-4)
    assert hb[KR]["extra_paid"] == h["extra_paid_vs_cheapest"]
    assert any("under 1.00" in n for n in d["notes"])


def test_the_rule_block_carries_the_benchmarks_thresholds(client, data):
    # the page words '10%', 'half' and '1%' from here (app/etl/purchase/prices.py), never its own copy
    want = {"usual_share_pct": 10.0, "low_ratio_pct": 50.0, "spread_min_pct": 1.0}
    assert want == pytest.approx({"usual_share_pct": P.USUAL_SHARE * 100, "low_ratio_pct": P.LOW_RATIO * 100,
                                  "spread_min_pct": P.SPREAD_MIN * 100})
    assert prices(client, Month=AUG)["rule"] == want
    assert prices(client, Month="ALL", Plant="HS01")["rule"] == want      # a hospital without purchases too
    # the Products tab's list carries the same block on every page, with its facets and on an empty cut
    for params in ({"Month": AUG}, {"Month": AUG, "page": 1, "page_size": 1}, {"Month": AUG, "facets": "1"},
                   {"Month": AUG, "Vendor": SDH}, {"Month": "ALL", "Plant": "HS01"}):
        d = get(client, "/spend/prices/molecules", **params)
        assert d["rule"] == want, params
    assert get(client, "/spend/prices/molecules", Month="ALL", Plant="HS01")["total"] == 0
    # the usual cheapest generic: no purchase is flagged low and no spread is reported there
    want_generic = {"usual_share_pct": 10.0, "low_ratio_pct": None, "spread_min_pct": None}
    assert generic(client, Month=AUG)["rule"] == want_generic
    assert generic(client, Month=AUG, Plant="HS01")["rule"] == want_generic


def test_the_notes_word_the_usual_cheapest_as_the_pages_method_does(client, data):
    # Counted up from the cheapest purchase (prices.usual_cheapest), so the price named can itself
    # hold under the share: never "the lowest price at which 10% was bought" (spend insights.tsx
    # usualCheapestRule). The share is the build's own constant, printed as the rule block gives it.
    share = f"{P.USUAL_SHARE * 100:g}%"
    assert PI.USUAL_SHARE_TEXT == share == "10%"
    rule = (f"the unit price reached once at least {share} of the product's units have been counted, cheapest "
            f"purchase first, so one odd cheap purchase cannot set it.")
    generic_rule = (f"the usual cheapest generic price (the generic price reached once at least {share} of that "
                    f"month's generic units have been counted, cheapest purchase first)")
    assert PI.NOTE_PRICE_VALUES in prices(client, Month=AUG)["notes"] and rule in PI.NOTE_PRICE_VALUES
    assert PI.NOTE_GENERIC_VALUES in generic(client, Month=AUG)["notes"] and generic_rule in PI.NOTE_GENERIC_VALUES
    assert all("at which at least" not in n for n in (PI.NOTE_PRICE_VALUES, PI.NOTE_GENERIC_VALUES))


def test_a_hospital_without_purchases_has_an_empty_low_list(client, checks):
    d = prices(client, Month=BOTH, Plant="HS01")
    assert d["low_prices"] == {"rows": 0, "value": 0.0, "products": 0, "items": []}
    assert (d["headline"]["low_rows"], d["headline"]["low_value"], d["molecule"], d["molecules"]) == (0, 0.0, None, [])


def test_every_purchase_to_check_is_in_the_paged_list(client, checks, monkeypatch):
    # The overview lists the first TOP_N groups; /spend/prices/low pages through every one, in the
    # same order with the same figures: the three groups of the low-list test above. Sums: qty
    # 2 + 1 + 10 = 13, value 6,000 + 4,000 + 2,000 = 12,000, under the cheapest 14,000 + 6,000 +
    # 3,000 = 23,000, lines 2 + 1 + 1 = 4.
    monkeypatch.setattr(PI, "TOP_N", 2)
    PI.clear_caches()
    d = prices(client, Month=BOTH)
    full = get(client, "/spend/prices/low", Month=BOTH)
    assert (d["low_prices"]["rows"], full["total"]) == (3, 3)
    assert full["data"][:2] == d["low_prices"]["items"] and len(d["low_prices"]["items"]) == 2
    assert [(r["plant"], r["material"], r["vendor_name"], r["below_cheapest"]) for r in full["data"]] == [
        (DR, "105002", VARDHMAN, 14000.0), (KR, "105002", SDH, 6000.0), (DR, "205001", SDH, 3000.0)]
    assert list(full["data"][0]) == PI.LOW_COLUMNS
    assert full["sums"] == {"qty": 13.0, "value": 12000.0, "below_cheapest": 23000.0, "lines": 4}
    assert full["rule"] == d["rule"]
    # paged, filtered by its facets, and exported 5,000 rows a page at most
    assert get(client, "/spend/prices/low", Month=BOTH, page=1, page_size=1)["data"][0]["plant"] == KR
    f = get(client, "/spend/prices/low", Month=BOTH, facets="1")
    assert {x["field"] for x in f["facets"]} >= {"plant", "minor_group", "vendor_name"}
    one = get(client, "/spend/prices/low", **{"Month": BOTH, "f.minor_group": "ONDANSETRON"})
    assert (one["total"], one["sums"]["value"]) == (1, 2000.0)
    assert client.get("/spend/prices/low", params={"Month": BOTH, "page_size": 5001}).status_code == 400
    # the page filters cut it as they cut the overview: August's own cheapest, KR's own
    assert get(client, "/spend/prices/low", Month=AUG)["total"] == prices(client, Month=AUG)["low_prices"]["rows"] == 2
    assert get(client, "/spend/prices/low", Month="ALL", Plant=KR)["total"] == 1
    # nothing to check: an empty list with zero sums, never an error
    empty = get(client, "/spend/prices/low", Month=JUL)
    assert (empty["total"], empty["data"], empty["sums"]["value"]) == (0, [], 0.0)
    assert get(client, "/spend/prices/low", Month=BOTH, Plant="HS01")["total"] == 0
    assert client.get("/spend/prices/low", params={"Plant": "NOWHERE"}).status_code == 404
    assert client.get("/spend/prices/low", params={"Month": "2025-01"}).status_code == 400


# A product whose strength the purchase build writes in its own number form, '4.5e+06IU' (the
# key keeps that spelling, lower-case 'e' included), bought by KR at 2,000 and DR at 3,000 a vial.
COLI = ("COLISTIMETHATE|4.5e+06IU|INJ|", "COLISTIMETHATE", "4.5e+06IU", "INJ")
COLI_BRANDS = {"106001": ("COLISTIN 4.5MIU INJ VIAL", "COLISTIMETHATE", "CIPLA", COLI)}
COLI_ROWS = [(AUG, KR, "106001", "50790", "ok", 10, 20000.0), (AUG, DR, "106001", "52366", "ok", 10, 30000.0)]


def test_a_product_key_is_found_as_the_build_spells_it(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "purchase"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_prices(tmp_path / "purchase", COLI_ROWS, COLI_BRANDS, VENDORS)
    write_hospital_map(tmp_path / "reference")
    PI.clear_caches()
    # 20 vials; the usual cheapest is KR's 2,000 (10 >= 2); DR paid 30,000 - 20,000 = 10,000 above it
    for sent in (COLI[0], COLI[0].upper(), COLI[0].lower()):
        d = prices(client, Month=AUG, Molecule=sent)
        assert d["molecule"]["molecule_key"] == d["scope"]["molecule"] == COLI[0], sent
        assert (d["molecule"]["cheapest_price"], d["molecule"]["extra_paid"]) == (2000.0, 10000.0), sent
        assert not any("was not bought" in n for n in d["notes"]), sent
    d = prices(client, Month=AUG, Molecule="COLISTIMETHATE|9e+06IU|INJ|")
    assert d["molecule"] is None and "COLISTIMETHATE|9e+06IU|INJ| was not bought at a comparable price" in \
        " ".join(d["notes"])
    PI.clear_caches()


# ---------- generic vs branded ----------

def test_generic_headline_by_hand(client, data):
    h = generic(client, Month=AUG)["headline"]
    assert (h["value"], h["branded_value"], h["generic_value"]) == (335350.0, 315150.0, 20200.0)
    assert h["branded_pct"] == pytest.approx(315150 / 335350 * 100, abs=1e-4)
    assert h["generic_pct"] == pytest.approx(20200 / 335350 * 100, abs=1e-4)
    assert (h["extra_vs_generic"], h["compared_value"], h["branded_not_compared"]) == (190000.0, 265150.0, 50000.0)
    assert h["extra_vs_generic_pct"] == pytest.approx(190000 / 265150 * 100, abs=1e-4)
    assert (h["lines"], h["branded_lines"], h["brands"], h["generic_brands"], h["branded_brands"], h["hospitals"]) == \
        (6, 4, 5, 2, 3, 2)


def test_generic_months_hospitals_segments_and_sources(client, data):
    d = generic(client, Month=AUG)
    assert [(m["label"], m["item_master"], m["generic_value"], m["branded_value"], m["extra_vs_generic"])
            for m in d["by_month"]] == [("May-26", False, 4000.0, 60000.0, 48000.0),
                                        ("Aug-26", True, 20200.0, 315150.0, 190000.0)]
    assert [(h["plant"], h["rank"], h["branded_value"], h["generic_value"], h["extra_vs_generic"])
            for h in d["by_hospital"]] == [(KR, 1, 180000.0, 20200.0, 130000.0), (DR, 2, 135150.0, 0.0, 60000.0)]
    seg = by_key(d["by_segment"], "key")
    assert (seg["onco"]["value"], seg["onco"]["branded_value"], seg["onco"]["extra_vs_generic"]) == \
        (335000.0, 315000.0, 190000.0)
    # CALPOL was bought below its generic: compared, nothing extra
    assert (seg["non-onco"]["compared_value"], seg["non-onco"]["extra_vs_generic_pct"]) == (150.0, 0.0)
    assert {s["key"]: (s["value"], s["lines"]) for s in d["by_source"]} == {
        "override": (50000.0, 1), "originator_list": (265150.0, 3), "default": (20200.0, 2)}


def test_compliance_split_uses_each_months_own_item_master(client, data):
    c = generic(client, Month=AUG)["compliance"]
    assert (c["months"], c["months_without_item_master"], c["available"]) == ([AUG], [], True)
    g, b = c["by_class"]
    assert (g["class"], g["value"], b["class"], b["value"]) == ("generic", 20200.0, "branded", 315150.0)
    assert {f["key"]: f["value"] for f in g["flags"]} == {"FORMULARY": 200.0, "NON FORMULARY": 0.0,
                                                          "OUT OF FORMULARY": 20000.0, "NOT IN MASTER": 0.0}
    assert {f["key"]: f["value"] for f in b["flags"]} == {"FORMULARY": 265150.0, "NON FORMULARY": 50000.0,
                                                          "OUT OF FORMULARY": 0.0, "NOT IN MASTER": 0.0}
    assert {t["key"]: t["value"] for t in b["tiers"]} == {"preferred": 150.0, "scheme_only": 0.0,
                                                          "restricted": 265000.0, "other": 0.0}
    assert b["flags"][1]["share_pct"] == pytest.approx(50000 / 315150 * 100, abs=1e-4)
    d = generic(client, Month="ALL")                       # May has no Item Master
    assert (d["compliance"]["months"], d["compliance"]["months_without_item_master"]) == ([AUG], [MAY])
    assert any("covers Aug-26 only — May-26 has no Item Master" in n for n in d["notes"])
    d = generic(client, Month=MAY)
    assert not d["compliance"]["available"] and d["compliance"]["by_class"][1]["value"] == 0.0
    assert any("May-26 has none" in n for n in d["notes"])


def test_top_branded_products_with_a_cheaper_generic(client, data):
    top = generic(client, Month=AUG)["top_molecules"]
    assert len(top) == 1                                   # CALPOL saved nothing; VECTIBIX had no generic
    t = top[0]
    assert (t["molecule_key"], t["brands"], t["companies"], t["plants"]) == \
        (CETUX[0], ["ERBITUX 100MG INJ VIAL"], ["Merck"], 2)
    assert (t["extra_vs_generic"], t["compared_value"], t["branded_value"], t["share_pct"]) == \
        (190000.0, 265000.0, 265000.0, 100.0)
    assert t["extra_vs_generic_pct"] == pytest.approx(190000 / 265000 * 100, abs=1e-4)
    assert (t["generic_material"], t["generic_desc"], t["generic_price"], t["generic_period"]) == \
        ("101002", "CETUXA 100MG INJ VIAL", 5000.0, AUG)
    t = generic(client, Month="ALL")["top_molecules"][0]   # + May's 48,000; the latest month's generic
    assert (t["extra_vs_generic"], t["compared_value"], t["generic_price"], t["generic_period"]) == \
        (238000.0, 325000.0, 5000.0, AUG)


def test_generic_definition_is_estimated(client, data):
    d = generic(client, Month=AUG)
    assert d["estimated"] is True
    df = d["definition"]
    assert (df["text"], df["estimated"], df["label"], df["originator_companies"]) == \
        (DEFINITION, True, "Estimated definition", 43)
    assert df["overrides"][0]["material"] == "100714" and set(df["class_sources"]) == set(Gn.CLASS_SOURCES)
    (data / "purchase" / C.VALIDATION_FILE).unlink()
    PI.clear_caches()
    assert generic(client, Month=AUG)["definition"]["text"] == Gn.DEFINITION
    # Estimated: the app's own reading, with nothing for HCG to confirm
    assert "Estimated definition" in Gn.DEFINITION and "until HCG" not in Gn.DEFINITION


def test_generic_brand_filter_and_a_hospital_without_purchases(client, data):
    h = generic(client, Month=AUG, Material="101001.0")["headline"]
    assert (h["value"], h["branded_value"], h["extra_vs_generic"], h["brands"]) == (265000.0, 265000.0, 190000.0, 1)
    d = generic(client, Month=AUG, Plant="HS01")
    h = d["headline"]
    assert (h["value"], h["branded_pct"], h["extra_vs_generic_pct"]) == (0.0, None, None)
    assert d["top_molecules"] == [] and any("has no purchase" in n for n in d["notes"])
    h = prices(client, Month=AUG, Plant="HS01")["headline"]
    assert (h["molecules"], h["extra_paid_vs_cheapest"], h["extra_pct"], h["extra_paid_vs_network_cheapest"]) == \
        (0, 0.0, None, 0.0)


# ---------- arguments, auth, missing data ----------

ROUTES = ("/spend/prices/overview", "/spend/prices/molecules", "/spend/generic/overview")


def test_bad_parameters_are_a_400_and_unknown_plant_a_404(client, data):
    for params in ({"Month": "2026-13"}, {"Month": "2025-01"}, {"Segment": "bones"}, {"Category": "Consumables"},
                   {"Segment": "onco", "Category": "Other Drugs"}):
        for path in ROUTES:
            assert client.get(path, params=params).status_code == 400, (path, params)
    for path in ROUTES:
        assert client.get(path, params={"Plant": "NOWHERE"}).status_code == 404, path


def test_every_route_needs_a_login(anon_client, data):
    for path in ROUTES:
        assert anon_client.get(path).status_code == 401, path


def test_missing_tables_are_a_503_with_the_build_hint(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PURCHASE_DATA_DIR", str(tmp_path / "nothing"))
    PI.clear_caches()
    for path in ROUTES:
        r = client.get(path)
        assert r.status_code == 503 and "python -m app.etl.purchase" in r.json()["detail"], path
    # /spend/meta's purchase months stay served, empty
    assert PI.months_meta() == {"prices_months": [], "generic_months": [], "purchase_built_at": None}


def test_the_page_meta_carries_each_sections_months(client, data):
    # /spend/meta gives the page these months, so it need not fetch an overview to learn them
    m = PI.months_meta()
    assert m["prices_months"] == prices(client)["months_available"]
    assert m["generic_months"] == generic(client)["months_available"]
    assert m["purchase_built_at"] == prices(client)["data_as_of"]


def test_a_table_from_an_older_build_is_a_503(client, data):
    p = data / "purchase" / f"{C.PRICES}.parquet"
    pd.read_parquet(p).drop(columns=["price_status"]).to_parquet(p, index=False)
    r = client.get("/spend/prices/overview")
    assert r.status_code == 503 and "older version" in r.json()["detail"]
    assert client.get("/spend/generic/overview").status_code == 200          # the other table still serves


def test_a_rebuild_is_served_without_a_restart(client, data):
    assert prices(client, Month=AUG)["headline"]["extra_paid_vs_cheapest"] == 192050.0
    p = data / "purchase" / f"{C.PRICES}.parquet"
    df = pd.read_parquet(p)
    df.loc[0, ["value", "unit_price_min", "unit_price_max"]] = [190000.0, 19000.0, 19000.0]   # p1 at 19,000
    df.to_parquet(p, index=False)
    assert prices(client, Month=AUG)["headline"]["extra_paid_vs_cheapest"] == 202050.0


def _keys(obj) -> set:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


def test_no_identifier_fields_in_any_response(client, data):
    bad = ("gr_no", "gr_number", "po_no", "po_number", "invoice", "reference", "uhid", "patient", "bill", "gr_date")
    for path in ROUTES:
        keys = _keys(get(client, path, Month="ALL"))
        assert not [k for k in keys if any(b in k.lower() for b in bad)], path


# ---------- the committed tables (python -m app.etl.purchase) ----------

DATA = Path(settings.PURCHASE_DATA_DIR)
committed = pytest.mark.skipif(not all((DATA / f"{t}.parquet").exists() for t in (C.PRICES, C.GENERIC)),
                               reason="purchase tables not built — run python -m app.etl.purchase")


@pytest.fixture()
def report():
    rep = json.loads((DATA / C.VALIDATION_FILE).read_text())
    if AUG not in rep.get("months", {}):
        pytest.skip("2026-08 not built")
    PI.clear_caches()
    return rep


@committed
def test_real_august_generic_vs_branded(client, report):
    d = generic(client, Month=AUG)
    h, rep = d["headline"], report["months"][AUG]["generic"]
    # read when the definition was chosen (originator list + the three overrides)
    assert (h["branded_value"], h["generic_value"], h["value"]) == (235638720.20, 220121589.88, 455760310.08)
    # the usual cheapest generic (prices.usual_cheapest): Rs 2,79,41,233.49 when it was the
    # lowest generic material's median price — one vial of CETUXA at 5,500 then priced every ERBITUX
    assert (h["extra_vs_generic"], h["compared_value"]) == (24702552.55, 37090389.80)
    assert h["branded_value"] == rep["by_class"]["branded"]["value"]
    assert h["extra_vs_generic"] == rep["cost_impact"]["extra_vs_generic"]
    for side in d["compliance"]["by_class"]:
        flags = {f["key"]: f["value"] for f in side["flags"] if f["value"]}
        tiers = {t["key"]: t["value"] for t in side["tiers"] if t["value"]}
        assert flags == pytest.approx(rep["formulary"][side["class"]], abs=0.01), side["class"]
        assert tiers == pytest.approx(rep["tiers"][side["class"]], abs=0.01), side["class"]
    top = d["top_molecules"]
    assert [t["molecule_key"] for t in top[:2]] == ["NIVOLUMAB|100MG|INJ|", "CETUXIMAB|100MG|INJ|"]
    assert top[1]["generic_desc"] == "LUPITUX 100MG INJ VIAL" and top[1]["extra_vs_generic"] == 5422521.0


@committed
def test_real_price_variation_matches_the_build_report(client, report):
    d = prices(client, Month=AUG)
    h, rep = d["headline"], report["months"][AUG]["prices"]
    # against the usual cheapest (Rs 8,98,82,175.13 against the plain minimum)
    assert (h["molecules"], h["molecules_with_spread"], h["extra_paid_vs_cheapest"]) == (1819, 800, 54642940.44)
    assert (h["low_rows"], h["low_value"]) == (40, 170535.95) == (d["low_prices"]["rows"], d["low_prices"]["value"])
    assert (rep["unusually_low"]["rows"], rep["unusually_low"]["value"]) == (40, 170535.95)
    bryxta = next(i for i in d["low_prices"]["items"] if i["material_desc"].startswith("BRYXTA 400MG"))
    assert (bryxta["plant"], bryxta["unit_price"], bryxta["cheapest_price"], bryxta["below_cheapest"]) == \
        ("HC35", 7500.0, 39000.0, 31500.0)
    assert (h["molecules"], h["molecules_with_spread"]) == (rep["products"], rep["products_with_spread"])
    assert (h["extra_paid_vs_cheapest"], h["compared_value"], h["value"]) == \
        (rep["extra_paid_vs_cheapest"], rep["compared"]["value"], rep["value"])
    # every month on its own cheapest = the build's month report
    for m in d["by_month"]:
        r = report["months"][m["period"]]["prices"]
        assert (m["extra_paid"], m["molecules"], m["molecules_with_spread"]) == \
            (r["extra_paid_vs_cheapest"], r["products"], r["products_with_spread"]), m["period"]
    # the hospital ranking adds up to the all-hospital figure
    assert sum(x["extra_paid"] for x in d["by_hospital"]) == pytest.approx(h["extra_paid_vs_cheapest"], abs=0.05)


@committed
def test_real_cetuximab_spread_and_the_biggest_vendor(client, report):
    d = prices(client, Month=AUG, Molecule="CETUXIMAB|100MG|INJ|")
    m = d["molecule"]
    # usual cheapest 8,100 (LUPITUX); the 5,100 vials are under 10% of the quantity, still the lowest
    assert (m["cheapest_price"], m["min_price"], m["max_price"], m["brands"], m["plants"]) == \
        (8100.0, 5100.0, 18938.1, 3, 15)
    erbitux = [p for p in m["points"] if p["material_desc"].startswith("ERBITUX")]
    assert (min(p["unit_price_min"] for p in erbitux), max(p["unit_price_max"] for p in erbitux)) == \
        (16920.6, 18938.1)
    v = d["vendors"][0]                                    # one supplier, eleven SAP codes
    # 961 products: its LIPODOX (liposomal doxorubicin) is a product of its own, not plain doxorubicin
    assert (v["vendor_name"], v["molecules"], v["value"]) == ("Vardhman Health Specialities", 961, 413168995.54)
    assert v["price_index"] == pytest.approx(1.1412, abs=5e-4)


@committed
def test_real_kr_against_its_own_cheapest_and_the_networks(client, report):
    d = prices(client, Month=AUG, Plant="HC05")
    h = d["headline"]
    assert (h["molecules"], h["extra_paid_vs_cheapest"]) == (698, 6676595.12)
    assert h["extra_paid_vs_network_cheapest"] == next(x["extra_paid"] for x in d["by_hospital"] if x["selected"])
    assert h["extra_paid_vs_network_cheapest"] >= h["extra_paid_vs_cheapest"]
