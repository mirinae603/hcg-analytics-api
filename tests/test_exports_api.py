# tests/test_exports_api.py — /export/meta, /export/sales and /export/grn.
#
# Both source tables are written here from invented rows: 8 sales_items rows (two hospitals,
# July and August, every formulary bucket, IP and OP, the three payers, a consumable with a
# blank cost price, amounts that need rounding) and 6 fact_grn lines (three plants, one missing
# from dim_plant, a receipt dated 31 July but posted in August, a blank expiry date). Each source
# also carries decoy identifier columns that must never come out. Every expected number below was
# worked out by hand from these rows (comments show the arithmetic). The last tests read the
# COMMITTED tables (skipped while a table is missing) and set the export beside the table itself.
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.api import exports as E
from app.core import table_query as tq
from app.core.config import settings
from app.etl.formulary import config as FC
from app.etl.ingest import FACT_GRN_COLUMNS

from tests.test_formulary_api import write_hospital_map

JUL, AUG = "2026-07", "2026-08"
KR, DR, SUN = "HC05", "HC06", "HS01"           # SUN is in the app's plant list but not in our dim_plant
HIS = {KR: ("KABHK", "HCG KR, Bangalore"), DR: ("KABHD", "HCG DR, Bangalore")}
# Column names no export may carry. patient_type is IP / OP — the kind of stay, not a patient —
# and is the one name the pattern lets through.
IDENTIFIER = re.compile(r"(gr_?no|po_?no|invoice|uhid|patient(?!_type$)|bill|doctor|batch)", re.I)
DECOYS = {"doctor_name": "DECOY", "uhid": "DECOY", "bill_no": "DECOY", "po_no": "DECOY", "gr_no": "DECOY",
          "invoice_no": "DECOY", "batch_code": "DECOY"}

# material -> (description, generic name, manufacturer, category)
ITEMS = {
    "101001": ("ERBITUX 100MG INJ VIAL", "CETUXIMAB", "MERCK", "Onco Drugs"),
    "101002": ("CETUXA 100MG INJ VIAL", "CETUXIMAB", "INTAS", "Onco Drugs"),
    "201001": ("PARA A 500MG TAB", "PARACETAMOL", "MICRO LABS", "Other Drugs"),
    "201002": ("PARA B 500MG TAB", "PARACETAMOL", "CIPLA", "Other Drugs"),
    "301001": ("SYRINGE 5ML", None, "BD", "Consumables"),
}
# id: period, plant, type, payer, material, class, group, cost, mrp, qty, lines, blank-cost lines, their MRP
SALES = {
    "s1": (AUG, KR, "IP", "CASH", "101001", "FORMULARYR", "A", 8000.0, 10000.0, 2, 2, 0, 0.0),
    "s2": (AUG, KR, "IP", "GOV", "101002", "OUT OF FORMULARY", "B", 3000.0, 4500.0, 1, 1, 0, 0.0),
    "s3": (AUG, KR, "OP", "CASH(TPA)", "201001", "NON FORMULARY", "C", 20.0, 50.0, 10, 3, 1, 15.0),
    # a consumable is "Consumables" whatever its group; its one line had no cost price
    "s4": (AUG, KR, "IP", "CASH", "301001", "NOT IN MASTER", "X", 0.0, 120.5, 5, 1, 1, 120.5),
    # rounding: cost 10.004 -> 10.00, MRP 25.006 -> 25.01, margin 15.01, units 4.0004 -> 4.000
    "s5": (AUG, DR, "OP", "CASH", "201002", "NOT IN MASTER", "X", 10.004, 25.006, 4.0004, 1, 0, 0.0),
    "s6": (AUG, DR, "IP", "GOV", "101001", "FORMULARYR", "A", 4000.0, 5000.0, 1, 1, 0, 0.0),
    "s7": (JUL, KR, "IP", "CASH", "101001", "FORMULARYR", "A", 12000.0, 15000.0, 3, 3, 0, 0.0),
    "s8": (JUL, DR, "OP", "CASH", "201001", "FORMULARYP", "A", 30.0, 60.0, 20, 2, 0, 0.0),
}
# Export order: month, hospital, item (then IP/OP, payer, class)
AUG_ORDER = ["s1", "s2", "s3", "s4", "s6", "s5"]
ALL_ORDER = ["s7", "s8"] + AUG_ORDER
# Aug, every hospital:
#   cost   8,000 + 3,000 + 20 + 0 + 10.00 + 4,000     = 15,030.00
#   MRP   10,000 + 4,500 + 50 + 120.50 + 25.01 + 5,000 = 19,695.51
#   margin 2,000 + 1,500 + 30 + 120.50 + 15.01 + 1,000 = 4,665.51 (= 19,695.51 - 15,030.00)
#   units  2 + 1 + 10 + 5 + 4 + 1 = 23; lines 2 + 1 + 3 + 1 + 1 + 1 = 9; blank-cost lines 2 (MRP 135.50)
AUG_SUMS = {"qty": 23.0, "cost": 15030.0, "mrp": 19695.51, "margin": 4665.51, "lines": 9,
            "cost_missing_lines": 2, "mrp_cost_missing": 135.5}

VARDHMAN, SDH = "Vardhman Health Specialities", "Speciality Drug House"
VENDORS = {"50790": VARDHMAN, "52366": SDH}
MAB, ANALGESIC = "MONOCLONAL ANTIBODY", "ANALGESICS"
# id: (year, month_num), plant, material, vendor code, qty, base price, MRP a unit, amount before
#     tax, tax %, tax, amount with tax, discount, MRP value, major group, order, receipt, expiry
GRN = {
    "g1": ((2026, 8), KR, "101001", "50790", 10, 18000.0, 22000.0, 180000.0, 5.0, 9000.0, 189000.0, 0.0,
           220000.0, MAB, "2026-07-28", "2026-08-03", "2028-01-31"),
    "g2": ((2026, 8), KR, "201001", "52366", 100, 2.0, 3.0, 200.0, 12.0, 24.0, 224.0, 10.0, 300.0, ANALGESIC,
           "2026-07-30", "2026-08-01", None),
    # received 31 July, posted 1 August: an August receipt (the month is the posting month)
    "g3": ((2026, 8), DR, "101002", "52366", 4, 5000.0, 6500.0, 20000.0, 5.0, 1000.0, 21000.0, 0.0, 26000.0,
           MAB, "2026-07-25", "2026-07-31", "2027-12-31"),
    "g4": ((2026, 8), KR, "101002", "52366", 1, 5000.0, 6500.0, 5000.0, 5.0, 250.0, 5250.0, 0.0, 6500.0, MAB,
           "2026-07-29", "2026-08-01", "2027-12-31"),
    # rounding: 50.0004 received -> 50.000, base price 1.4999 -> 1.50
    "g5": ((2026, 8), SUN, "201001", "52366", 50.0004, 1.4999, 3.0, 75.0, 12.0, 9.0, 84.0, 0.0, 150.0,
           ANALGESIC, "2026-08-05", "2026-08-10", "2028-06-30"),
    "g6": ((2026, 7), KR, "101001", "50790", 3, 20000.0, 22000.0, 60000.0, 5.0, 3000.0, 63000.0, 0.0,
           66000.0, MAB, "2026-07-01", "2026-07-15", "2028-01-31"),
}
# Export order: month, plant, receipt date, item -> KR on 1 Aug: g4 (101002) before g2 (201001),
# then g1 (3 Aug); DR g3; SUN g5
GRN_AUG_ORDER = ["g4", "g2", "g1", "g3", "g5"]
# Aug, every plant: qty 10 + 100 + 4 + 1 + 50 = 165; before tax 1,80,000 + 200 + 20,000 + 5,000 + 75
# = 2,05,275; tax 9,000 + 24 + 1,000 + 250 + 9 = 10,283; with tax 2,15,558; discount 10;
# MRP value 2,20,000 + 300 + 26,000 + 6,500 + 150 = 2,52,950
GRN_AUG_SUMS = {"gr_qty": 165.0, "total_amount_wo_tax": 205275.0, "tax_amount": 10283.0,
                "total_amount_with_tax": 215558.0, "discount_value": 10.0, "total_mrp_value": 252950.0}
PLANT_NAMES = {KR: "HCG KR, Bangalore", DR: "HCG DR, Bangalore"}


def sales_frame(keys=None) -> pd.DataFrame:
    rows = []
    for key in (keys or SALES):
        (period, plant, ptype, payer, mat, cls, grp, cost, mrp, qty, lines, blank, blank_mrp) = SALES[key]
        desc, generic, maker, cat = ITEMS[mat]
        rows.append({"period": period, "plant": plant, "hospital": HIS[plant][0], "hospital_name": HIS[plant][1],
                     "patient_type": ptype, "payer_bucket": payer, "material": mat, "material_desc": desc,
                     "generic_name": generic, "manufacturer": maker, "category": cat, "formulary_class": cls,
                     "formulary_group": grp, "cost": cost, "mrp": mrp, "qty": float(qty), "lines": lines,
                     "cost_missing_lines": blank, "mrp_cost_missing": blank_mrp, **DECOYS})
    return pd.DataFrame(rows)


def write_sales(folder: Path, df: pd.DataFrame | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = sales_frame(list(reversed(SALES))) if df is None else df     # stored out of order on purpose
    df.to_parquet(folder / f"{FC.SALES_ITEMS}.parquet", index=False)


def write_grn(folder: Path, keys=None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in (keys or list(reversed(GRN))):
        ((year, month), plant, mat, vendor, qty, base, unit_mrp, wo_tax, tax_pct, tax, with_tax, discount,
         mrp_value, major, po, gr, expiry) = GRN[key]
        rows.append({"plant": plant, "sloc": "S079", "material": mat, "material_desc": ITEMS[mat][0],
                     "vendor_code": vendor, "vendor_name": VENDORS[vendor], "gr_qty": float(qty),
                     "base_price": base, "net_price": base, "unit_mrp": unit_mrp, "total_amount_wo_tax": wo_tax,
                     "tax_pct": tax_pct, "tax_amount": tax, "total_amount_with_tax": with_tax,
                     "discount_value": discount, "total_mrp_value": mrp_value, "po_to_gr_tat": 3.0,
                     "pr_to_gr_tat": np.nan, "major_group": major, "minor_group": ITEMS[mat][1],
                     "po_date": pd.Timestamp(po), "gr_date": pd.Timestamp(gr),
                     "expiry_date": pd.Timestamp(expiry) if expiry else pd.NaT,
                     "posting_date": pd.Timestamp(year, month, 1), "year": year, "month_num": month,
                     "month": pd.Timestamp(year, month, 1).month_name()})
    df = pd.DataFrame(rows)[FACT_GRN_COLUMNS].assign(**{k: v for k, v in DECOYS.items() if k != "doctor_name"})
    df.to_parquet(folder / f"{E.GRN}.parquet", index=False)
    pd.DataFrame({"plant": list(PLANT_NAMES), "plant_name": list(PLANT_NAMES.values())}).to_parquet(
        folder / f"{E.PLANTS}.parquet", index=False)


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "formulary"))
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path / "curated"))
    monkeypatch.setattr(settings, "REFERENCE_DIR", str(tmp_path / "reference"))
    write_sales(tmp_path / "formulary")
    write_grn(tmp_path / "curated")
    write_hospital_map(tmp_path / "reference")
    E.clear_caches()
    yield tmp_path
    E.clear_caches()


def get(client, path, **params):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    return r.json()


def sales(client, **params):
    return get(client, "/export/sales", **params)


def grn(client, **params):
    return get(client, "/export/grn", **params)


def ids(rows: list) -> list:
    """Which fixture sales rows came back, in order (month, hospital, IP/OP, payer and item
    tell them apart)."""
    key = {v[:5]: k for k, v in SALES.items()}
    return [key.get((r["period"], r["plant"], r["patient_type"], r["payer_bucket"], r["material"])) for r in rows]


def facet(d: dict, field: str) -> list:
    f = next(x for x in d["facets"] if x["field"] == field)
    return [(v["value"], v["count"]) for v in f["values"]]


def scope(d: dict) -> dict:
    return {c["param"]: c["value"] for c in d["scope"]}


# ---------- sales ----------

def test_sales_rows_by_hand(client, data):
    d = sales(client, page_size=50)              # default Month: the latest, August
    assert d["total"] == 6 and ids(d["data"]) == AUG_ORDER
    assert d["sums"] == AUG_SUMS
    r = {x: row for x, row in zip(AUG_ORDER, d["data"])}
    assert list(r["s1"]) == [c.field for c in E.SALES_COLUMNS]       # every column, in export order
    assert (r["s1"]["hospital_name"], r["s1"]["generic_name"], r["s1"]["formulary_class"], r["s1"]["bucket_label"],
            r["s1"]["margin"], r["s1"]["lines"]) == ("HCG KR, Bangalore", "CETUXIMAB", "FORMULARYR", "Formulary",
                                                      2000.0, 2)
    assert [r[k]["bucket_label"] for k in ("s2", "s3", "s4", "s5")] == \
        ["Out of formulary", "Non-formulary", "Consumables", "Not in Item Master"]
    # the consumable: cost 0 for its blank cost price, its MRP beside it
    assert (r["s4"]["cost"], r["s4"]["margin"], r["s4"]["cost_missing_lines"], r["s4"]["mrp_cost_missing"],
            r["s4"]["generic_name"]) == (0.0, 120.5, 1, 120.5, None)
    assert (r["s5"]["cost"], r["s5"]["mrp"], r["s5"]["margin"], r["s5"]["qty"]) == (10.0, 25.01, 15.01, 4.0)
    assert all(round(x["mrp"] - x["cost"] - x["margin"], 6) == 0 for x in d["data"])


def test_sales_page_filters(client, data):
    # KR in August: s1-s4 -> cost 11,020, MRP 14,670.50, margin 3,650.50
    for plant in (KR, "KABHK", "HCG KR, Bangalore"):              # SAP code, HIS code, HIS name
        d = sales(client, Plant=plant)
        assert ids(d["data"]) == ["s1", "s2", "s3", "s4"]
        assert (d["sums"]["cost"], d["sums"]["mrp"], d["sums"]["margin"]) == (11020.0, 14670.5, 3650.5)
    # OP: s3 + s5 -> cost 30.00, MRP 75.01
    d = sales(client, Type="op")
    assert ids(d["data"]) == ["s3", "s5"] and (d["sums"]["cost"], d["sums"]["mrp"]) == (30.0, 75.01)
    assert ids(sales(client, Segment="consumables")["data"]) == ["s4"]
    assert ids(sales(client, Category="Consumables")["data"]) == ["s4"]
    assert ids(sales(client, Segment="onco")["data"]) == ["s1", "s2", "s6"]
    d = sales(client, Payer="GOV")
    assert ids(d["data"]) == ["s2", "s6"] and d["sums"]["mrp"] == 9500.0
    assert ids(sales(client, Month=JUL)["data"]) == ["s7", "s8"]
    d = sales(client, Month="ALL", page_size=50)
    assert ids(d["data"]) == ALL_ORDER and d["sums"]["mrp"] == 34755.51          # + July 15,000 + 60


def test_sales_scope_and_facets(client, data):
    d = sales(client, Plant="KABHK", Type="IP", Segment="onco", Payer="CASH", facets=1)
    assert scope(d) == {"Plant": KR, "Month": AUG, "Type": "IP", "Segment": "onco", "Payer": "CASH"}
    assert ids(d["data"]) == ["s1"]
    d = sales(client, facets=1)
    assert facet(d, "patient_type") == [("IP", 4), ("OP", 2)]
    assert facet(d, "category") == [("Onco Drugs", 3), ("Other Drugs", 2), ("Consumables", 1)]
    assert facet(d, "bucket_label") == [("Formulary", 2), ("Consumables", 1), ("Non-formulary", 1),
                                        ("Not in Item Master", 1), ("Out of formulary", 1)]
    assert facet(d, "payer_bucket") == [("CASH", 3), ("CASH(TPA)", 1), ("GOV", 2)]
    assert facet(d, "plant") == [(DR, 2), (KR, 4)]                   # by name: HCG DR before HCG KR
    assert facet(d, "period") == [(AUG, 6)]
    assert scope(sales(client, Month="ALL", facets=1))["Month"] == "ALL"
    assert scope(sales(client, Month=f"{JUL},{AUG}", facets=1))["Month"] == "ALL"   # every month = ALL
    # a facet narrows the rows and the sums, and keeps its own options
    d = get(client, "/export/sales", facets=1, **{"f.bucket_label": "Formulary"})
    assert ids(d["data"]) == ["s1", "s6"]
    assert (d["sums"]["cost"], d["sums"]["mrp"], d["sums"]["margin"], d["sums"]["lines"]) == \
        (12000.0, 15000.0, 3000.0, 3)
    assert len(facet(d, "bucket_label")) == 5 and facet(d, "patient_type") == [("IP", 2)]


def test_sales_empty_selections_are_empty_tables(client, data):
    for params in ({"Plant": SUN}, {"Plant": DR, "Type": "IP", "Payer": "CASH"}):
        d = sales(client, **params)
        assert (d["total"], d["data"]) == (0, [])
        assert d["sums"] == {"qty": 0.0, "cost": 0.0, "mrp": 0.0, "margin": 0.0, "lines": 0,
                             "cost_missing_lines": 0, "mrp_cost_missing": 0.0}


def test_sales_bad_params(client, data):
    assert client.get("/export/sales", params={"Plant": "ZZ99"}).status_code == 404
    for params in ({"Month": "2026-13"}, {"Month": "2026-05"}, {"Month": ","}, {"Type": "XX"},
                   {"Segment": "drugs"}, {"Category": "Lab"}, {"Payer": "CORP"},
                   {"Segment": "onco", "Category": "Consumables"}, {"f.vendor_name": "x"},
                   {"page": "two"}):
        r = client.get("/export/sales", params=params)
        assert r.status_code == 400, (params, r.text)
    assert "No sales data for 2026-05" in client.get("/export/sales", params={"Month": "2026-05"}).text


def test_sales_paging_search_sort_and_projection(client, data):
    # the CSV export: 4 rows a page, only the columns on screen, the same total on every page
    fields = "period,plant,material,cost,mrp"
    pages = [sales(client, page=p, page_size=4, fields=fields) for p in (0, 1, 2)]
    assert [p["total"] for p in pages] == [6, 6, 6] and [len(p["data"]) for p in pages] == [4, 2, 0]
    rows = pages[0]["data"] + pages[1]["data"]
    assert all(list(r) == fields.split(",") for r in rows)
    assert [(r["plant"], r["material"]) for r in rows] == [(SALES[k][1], SALES[k][4]) for k in AUG_ORDER]
    assert pages[0]["sums"] == AUG_SUMS                                      # sums are not projected away
    d = sales(client, sort_field="mrp", sort_order="desc")
    assert ids(d["data"]) == ["s1", "s6", "s2", "s4", "s3", "s5"]
    # categories sort alphabetically; ties keep the export order
    assert ids(sales(client, sort_field="category")["data"]) == ["s4", "s1", "s2", "s6", "s3", "s5"]
    d = sales(client, global_filter="para", search_fields="material_desc,generic_name")
    assert ids(d["data"]) == ["s3", "s5"] and d["sums"]["mrp"] == 75.01


def test_sales_without_blank_cost_columns(client, data):
    # a table built before the blank-cost counts: the two columns are simply not there
    write_sales(data / "formulary", sales_frame().drop(columns=list(E.SALES_OPTIONAL)))
    d = sales(client)
    assert "cost_missing_lines" not in d["data"][0] and "mrp_cost_missing" not in d["sums"]
    cols = [c["field"] for c in get(client, "/export/meta")["datasets"][0]["columns"]]
    assert cols == [c.field for c in E.SALES_COLUMNS if c.field not in E.SALES_OPTIONAL]


def test_sales_table_missing_a_column_is_a_clear_503(client, data):
    write_sales(data / "formulary", sales_frame().drop(columns=["mrp"]))
    r = client.get("/export/sales")
    assert r.status_code == 503 and "mrp" in r.json()["detail"] and E.REBUILD_SALES in r.json()["detail"]


def test_a_rebuilt_table_is_picked_up_without_a_restart(client, data):
    assert sales(client)["total"] == 6
    write_sales(data / "formulary", sales_frame(["s1", "s7"]))
    assert sales(client)["total"] == 1 and sales(client, Month="ALL")["total"] == 2


# ---------- GRN ----------

def grn_ids(rows: list) -> list:
    return [next(k for k, v in GRN.items() if (r["plant"], r["material"], r["gr_date"]) == (v[1], v[2], v[15]))
            for r in rows]


def test_grn_rows_by_hand(client, data):
    d = grn(client, page_size=50)                # default Month: the latest, August
    assert d["total"] == 5 and grn_ids(d["data"]) == GRN_AUG_ORDER
    assert d["sums"] == GRN_AUG_SUMS
    r = dict(zip(GRN_AUG_ORDER, d["data"]))
    assert list(r["g1"]) == [c.field for c in E.GRN_COLUMNS]
    assert {k: r["g1"][k] for k in ("month", "plant", "plant_name", "sloc", "material", "vendor_code",
                                    "vendor_name", "gr_qty", "base_price", "total_amount_wo_tax", "tax_pct",
                                    "po_date", "gr_date", "expiry_date")} == \
        {"month": AUG, "plant": KR, "plant_name": "HCG KR, Bangalore", "sloc": "S079", "material": "101001",
         "vendor_code": "50790", "vendor_name": VARDHMAN, "gr_qty": 10.0, "base_price": 18000.0,
         "total_amount_wo_tax": 180000.0, "tax_pct": 5.0, "po_date": "2026-07-28", "gr_date": "2026-08-03",
         "expiry_date": "2028-01-31"}
    assert (r["g3"]["month"], r["g3"]["gr_date"]) == (AUG, "2026-07-31")     # the posting month counts
    assert r["g2"]["expiry_date"] is None
    assert (r["g5"]["plant_name"], r["g5"]["gr_qty"], r["g5"]["base_price"]) == (None, 50.0, 1.5)


def test_grn_page_filters_facets_and_scope(client, data):
    # KR in August: g4 + g2 + g1 -> before tax 5,000 + 200 + 1,80,000 = 1,85,200 on 111 received
    d = grn(client, Plant=KR, facets=1)
    assert grn_ids(d["data"]) == ["g4", "g2", "g1"]
    assert (d["sums"]["total_amount_wo_tax"], d["sums"]["gr_qty"]) == (185200.0, 111.0)
    assert scope(d) == {"Plant": KR, "Month": AUG}
    assert grn_ids(grn(client, Plant="KABHK")["data"]) == ["g4", "g2", "g1"]
    d = grn(client, Month="ALL", facets=1, page_size=50)
    assert grn_ids(d["data"]) == ["g6"] + GRN_AUG_ORDER and scope(d)["Month"] == "ALL"
    assert facet(d, "month") == [(AUG, 5), (JUL, 1)]
    d = grn(client, facets=1)
    assert facet(d, "major_group") == [(MAB, 3), (ANALGESIC, 2)]
    assert facet(d, "vendor_name") == [(SDH, 4), (VARDHMAN, 1)]
    # Analgesics: g2 + g5 -> before tax 275
    d = get(client, "/export/grn", **{"f.major_group": ANALGESIC})
    assert grn_ids(d["data"]) == ["g2", "g5"] and d["sums"]["total_amount_wo_tax"] == 275.0
    d = get(client, "/export/grn", Month="ALL", **{"f.month": JUL})
    assert grn_ids(d["data"]) == ["g6"] and d["sums"]["total_amount_wo_tax"] == 60000.0


def test_grn_empty_and_bad_params(client, data):
    d = grn(client, Plant=DR, Month=JUL)                 # DR received nothing in July
    assert (d["total"], d["data"], d["sums"]["total_amount_wo_tax"]) == (0, [], 0.0)
    assert client.get("/export/grn", params={"Plant": "ZZ99"}).status_code == 404
    for params in ({"Month": "2026-13"}, {"Month": "2026-06"}, {"f.bucket_label": "Formulary"}):
        r = client.get("/export/grn", params=params)
        assert r.status_code == 400, (params, r.text)
    assert "No GRN data for 2026-06" in client.get("/export/grn", params={"Month": "2026-06"}).text


def test_a_page_is_at_most_the_export_csvs_page(client, data):
    # The Export CSV asks 5,000 rows a page (hcg-analytics-ui exportCsv.ts EXPORT_PAGE); one page
    # of more, a page_size of 0 or under, or a page under 0 is a 400, not the whole table at once
    # (a negative size used to slice from the far end: page_size=-5 gave all but the last 5 rows).
    for path in ("/export/grn", "/export/sales"):
        assert get(client, path, Month="ALL", page_size=tq.EXPORT_MAX_PAGE)["total"] > 0, path
        for params in ({"page_size": tq.EXPORT_MAX_PAGE + 1}, {"page_size": 0}, {"page_size": -5},
                       {"page": -1}, {"page": -3, "page_size": 2}):
            r = client.get(path, params={"Month": "ALL", **params})
            assert r.status_code == 400 and "page" in r.json()["detail"], (path, params, r.text)


def test_grn_export_walk_and_projection(client, data):
    fields = "month,plant,material,total_amount_wo_tax"
    pages = [grn(client, Month="ALL", page=p, page_size=4, fields=fields) for p in (0, 1)]
    rows = pages[0]["data"] + pages[1]["data"]
    assert [p["total"] for p in pages] == [6, 6] and all(list(r) == fields.split(",") for r in rows)
    assert sum(r["total_amount_wo_tax"] for r in rows) == pages[0]["sums"]["total_amount_wo_tax"] == 265275.0
    d = grn(client, sort_field="total_amount_wo_tax", sort_order="desc")
    assert grn_ids(d["data"]) == ["g1", "g3", "g4", "g2", "g5"]


def test_grn_is_held_once_and_paged_without_a_copy(data):
    one, two = E._grn(), E._grn()
    assert one.frame is two.frame and one.months == (JUL, AUG)
    every = E.GrnScope(plant=None, months=one.months)
    assert E._cut(one.frame, one.months, every, "month") is one.frame
    assert all(isinstance(one.frame[c.field].dtype, pd.CategoricalDtype) for c in E.GRN_COLUMNS if c.kind != "inr"
               and c.kind != "num")


# ---------- meta, missing tables, identifiers ----------

def test_meta_lists_both_datasets(client, data):
    m = get(client, "/export/meta")
    s, g = m["datasets"]
    assert (s["key"], s["endpoint"], s["available"], s["months"], s["rows_by_month"], s["rows"], s["latest"]) == \
        ("sales", "/export/sales", True, [JUL, AUG], {JUL: 2, AUG: 6}, 8, AUG)
    assert (g["key"], g["endpoint"], g["available"], g["months"], g["rows_by_month"], g["rows"]) == \
        ("grn", "/export/grn", True, [JUL, AUG], {JUL: 1, AUG: 5}, 6)
    assert s["month_labels"] == {JUL: "Jul-26", AUG: "Aug-26"}
    assert [c["field"] for c in s["columns"]] == [c.field for c in E.SALES_COLUMNS]
    assert [c["field"] for c in g["columns"]] == [c.field for c in E.GRN_COLUMNS]
    kinds = {c["field"]: c["kind"] for c in s["columns"] + g["columns"]}
    assert (kinds["mrp"], kinds["qty"], kinds["material"], kinds["gr_date"], kinds["tax_pct"]) == \
        ("inr", "num", "text", "date", "num")
    assert {c["kind"] for c in s["columns"] + g["columns"]} <= {"inr", "num", "text", "date"}
    assert s["params"] == ["Plant", "Month", "Type", "Segment", "Payer"] and g["params"] == ["Plant", "Month"]
    assert [x["key"] for x in s["options"]["Segment"]] == ["onco", "non-onco", "consumables"]
    assert s["source"] and g["source"] and s["note"] and g["note"]
    assert any("No patient, bill, invoice, GR or PO numbers" in n for n in m["notes"])
    assert m["data_as_of"] is not None


def test_missing_tables(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "FORMULARY_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(settings, "CURATED_DIR", str(tmp_path / "empty"))
    E.clear_caches()
    r = client.get("/export/sales")
    assert r.status_code == 503 and r.json()["detail"] == E.NOT_BUILT_SALES
    r = client.get("/export/grn")
    assert r.status_code == 503 and r.json()["detail"] == E.NOT_BUILT_GRN
    m = get(client, "/export/meta")                     # meta still answers, with nothing in either
    for ds, msg in zip(m["datasets"], (E.NOT_BUILT_SALES, E.NOT_BUILT_GRN)):
        assert (ds["available"], ds["months"], ds["rows_by_month"], ds["rows"], ds["latest"]) == \
            (False, [], {}, 0, None)
        assert msg in ds["note"] and ds["columns"]
    assert m["data_as_of"] is None
    E.clear_caches()


def test_one_table_missing_leaves_the_other(client, data):
    (data / "curated" / f"{E.GRN}.parquet").unlink()
    m = get(client, "/export/meta")
    assert [d["available"] for d in m["datasets"]] == [True, False]
    assert sales(client)["total"] == 6 and client.get("/export/grn").status_code == 503


def test_no_export_carries_an_identifier(client, data):
    names = set()
    for path, params in (("/export/sales", {"Month": "ALL"}), ("/export/grn", {"Month": "ALL"})):
        d = get(client, path, facets=1, page_size=50, **params)
        names |= {k for r in d["data"] for k in r} | set(d["sums"]) | {f["field"] for f in d["facets"]}
        # the decoys in the source never come out, as a column or as a value
        assert "DECOY" not in str(d["data"])
    names |= {c["field"] for ds in get(client, "/export/meta")["datasets"] for c in ds["columns"]}
    assert "patient_type" in names and "po_date" in names
    assert not [n for n in names if IDENTIFIER.search(n)]
    assert not [c.field for c in E.SALES_COLUMNS + E.GRN_COLUMNS if IDENTIFIER.search(c.field)]


def test_exports_need_a_login(anon_client, data):
    for path in ("/export/meta", "/export/sales", "/export/grn"):
        assert anon_client.get(path).status_code == 401


# ---------- the committed tables ----------

def _real(name: Path):
    if not name.exists():
        pytest.skip(f"{name.name} is not built")
    E.clear_caches()


def test_real_grn_august_at_kr(client):
    p = Path(settings.CURATED_DIR) / f"{E.GRN}.parquet"
    _real(p)
    t = pd.read_parquet(p, columns=["plant", "year", "month_num", "total_amount_wo_tax"])
    t = t[(t["plant"] == KR) & (t["year"] == 2026) & (t["month_num"] == 8)]
    d = grn(client, Plant=KR, Month=AUG, page_size=1)
    # HCG KR, August 2026: 3,560 receipt lines, Rs 16,59,52,603.07 before tax (read 2026-10-31)
    assert d["total"] == len(t) == 3560
    assert d["sums"]["total_amount_wo_tax"] == pytest.approx(float(t["total_amount_wo_tax"].sum()), abs=0.01)
    assert d["sums"]["total_amount_wo_tax"] == pytest.approx(165952603.07, abs=0.01)
    m = get(client, "/export/meta")["datasets"][1]
    assert m["rows"] == 383763 and m["months"][0] == "2025-12" and m["rows_by_month"][AUG] == 42422
    E.clear_caches()


def test_real_grn_export_walk(client):
    p = Path(settings.CURATED_DIR) / f"{E.GRN}.parquet"
    _real(p)
    fields = "month,plant,gr_date,material,total_amount_wo_tax"
    first = grn(client, Plant=KR, Month=AUG, page_size=1000, fields=fields)
    rows, page = list(first["data"]), 1
    while len(rows) < first["total"]:
        d = grn(client, Plant=KR, Month=AUG, page=page, page_size=1000, fields=fields)
        assert d["total"] == first["total"] and d["data"]
        rows += d["data"]
        page += 1
    assert len(rows) == 3560 and page == 4
    assert sum(r["total_amount_wo_tax"] or 0 for r in rows) == pytest.approx(first["sums"]["total_amount_wo_tax"],
                                                                             abs=0.01)
    keys = [(r["gr_date"], r["material"]) for r in rows]
    assert keys == sorted(keys)                          # one plant, one month: date then item
    E.clear_caches()


def test_real_sales_rows_equal_the_table(client):
    p = Path(settings.FORMULARY_DATA_DIR) / f"{FC.SALES_ITEMS}.parquet"
    _real(p)
    t = pd.read_parquet(p, columns=["period", "cost", "mrp"])
    aug = t[t["period"].astype(str) == AUG]
    if aug.empty:
        pytest.skip("sales_items holds no Aug-26")
    d = sales(client, Month=AUG, page_size=1)
    assert d["total"] == len(aug)
    assert d["sums"]["mrp"] == pytest.approx(float(aug["mrp"].round(2).sum()), abs=0.01)
    assert d["sums"]["cost"] == pytest.approx(float(aug["cost"].round(2).sum()), abs=0.01)
    assert sales(client, Month="ALL", page_size=1)["total"] == len(t)
    m = get(client, "/export/meta")["datasets"][0]
    assert m["rows_by_month"][AUG] == len(aug) and m["rows"] == len(t)
    E.clear_caches()
