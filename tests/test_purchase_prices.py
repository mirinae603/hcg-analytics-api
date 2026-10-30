# tests/test_purchase_prices.py — price variation (app/etl/purchase/prices.py) on tiny made-up data.
#
# The unit price is HCG's price after discount (Total Amount without tax / GR Qty); a line's
# price is compared only when it is a real per-unit price: free and credit lines, token prices
# (OPDYTA booked at Rs 0.01) and lines priced per strip or pack against a per-unit brand are
# kept in the totals but left out of the comparisons. Product keys are checked on real HCG
# descriptions and Item Master minor groups (product names only). The build runs through the
# same build.run() the real build uses, with every directory a tmp_path. The last section
# checks the committed table against figures read when the rule was chosen (skipped until
# `python -m app.etl.purchase` has published).
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.etl.purchase import build as B
from app.etl.purchase import config as C
from app.etl.purchase import grn as G
from app.etl.purchase import prices as P
from app.etl.purchase.masters import MainSheet, OofMaster
from app.etl.spend.masters import BrandList, PricedBrand

AUG, JUL, DEC = "2026-08", "2026-07", "2025-12"


# --- made-up data -----------------------------------------------------------------------------
def line(period, plant, material, qty, total, mrp=None, *, tax=5.0, vendor="300100",
         vendor_name=None, desc=None, minor="MOLECULE", maker="MAKER", base=None):
    """One GRN line as grn.read_grn_files gives it (no identifiers exist there). `total` is
    Total Amount without tax; Base Price defaults to the price after discount; each supplier
    code names its own supplier unless vendor_name says otherwise."""
    unit = total / qty if qty else 0.0
    return {"period": period, "plant": plant, "sloc": "S124", "material": material,
            "material_desc": desc or f"ITEM {material}", "vendor_code": vendor,
            "vendor_name": vendor_name or f"DISTRIBUTOR {vendor}", "mvt_type": "101", "gr_date": pd.Timestamp(f"{period}-04"),
            "gr_qty": float(qty), "uom": "EA", "base_price": unit if base is None else base,
            "discount_value": 0.0, "net_price": unit, "total_wo_tax": float(total), "tax_pct": tax,
            "unit_mrp": mrp, "manufacturer": maker, "material_type": "ZNOC",
            "minor_category": "DRUG", "major_group": "DRUGS", "minor_group": minor,
            "grn_formulary": "FORMULARY"}


def frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("period", "plant", "material", "material_desc", "vendor_code", "minor_group", "manufacturer"):
        df[c] = df[c].astype("str")
    df["unit_mrp"] = df["unit_mrp"].astype("float64")
    return df


def item_master(rows) -> pd.DataFrame:
    """(material, flag, tier, desc, minor group) -> masters.item_master_frame rows."""
    recs = [{"material": m, "formulary_class": ("FORMULARY" + (t or "")) if f == "FORMULARY" else f,
             "formulary_flag": f, "tier": t, "material_desc": d, "minor_group": g, "manufacturer": "MAKER"}
            for m, f, t, d, g in rows]
    return pd.DataFrame(recs, columns=["material", "formulary_class", "formulary_flag", "tier",
                                       "material_desc", "minor_group", "manufacturer"])


def main_sheet(list_mrp=None) -> MainSheet:
    prices = {m: PricedBrand(material=m, material_desc=None, price=mrp / 2, mrp=mrp, gst=5.0, category="P")
              for m, mrp in (list_mrp or {}).items()}
    cash = BrandList(path=Path("main.xlsx"), sheet="CASH&TPA", fy=(2026, 2027))
    blocks = pd.DataFrame(columns=["material", "material_desc", "block_id", "p_rank"])
    return MainSheet(path=Path("main.xlsx"), cash=cash, prices=prices, blocks=blocks)


SCOPE = C.Scope(mvt_keep=frozenset({"101"}), series_keep=frozenset({"1", "2"}), sloc=frozenset({"S125"}),
                plant_prefix=("BH",), plant=frozenset({"HM03"}))


def make_inputs(rows, item_masters=None, list_mrp=None, reference=None) -> B.Inputs:
    """build.Inputs around made-up lines: what the prices and generic builders read."""
    df = frame(rows)
    lines, funnel = G.apply_scope(df, SCOPE)
    periods = sorted(df["period"].unique())
    files = [G.GrnFileReport(file=f"GRN {p}.xlsx", sheet="GRN", period=p, rows=int((df["period"] == p).sum()),
                             lines=int((df["period"] == p).sum()), blank_rows=0, blank_material={},
                             mvt={}, returns=0, dates={"date": 1}, value=0.0, qty=0.0, source_size=1,
                             source_mtime_ns=1) for p in periods]
    ims = item_masters or {}
    return B.Inputs(scope=SCOPE, grn=G.GrnRead(folder=None, frame=df, files=files), lines=lines,
                    funnel=funnel, periods=periods, item_masters=ims,
                    item_master_files={p: f"IM {p}.xlsx" for p in ims}, hcg={},
                    master=OofMaster(name="master-2026-27.xlsb", sheet="Sheet1", fy=(2026, 2027)),
                    main=main_sheet(list_mrp), email={}, reference=reference, log=lambda *_: None)


# --- unit price and the product -------------------------------------------------------------
def test_the_unit_price_is_after_discount():
    df = frame([line(AUG, "HC05", "200010", 2, 180.0, 210.0, base=100.0),     # Rs 10 off per unit
                line(AUG, "HC05", "200010", 3, 100.0, 210.0),
                line(AUG, "HC05", "200010", 0, 50.0, 210.0)])
    assert P.unit_prices(df).tolist()[:2] == [90.0, 33.3333]
    assert pd.isna(P.unit_prices(df).iloc[2])


@pytest.mark.parametrize("minor, desc, key", [
    ("CETUXIMAB", "ERBITUX 100MG INJ VIAL", "CETUXIMAB|100MG|INJ|"),
    ("TRAMADOL HCL (SCH H1)", "COPTRADOL 50MG/ML INJ 2ML AMP", "TRAMADOL|100MG|INJ|"),   # per mL x 2 mL
    ("TRAMADOL HYDROCHLORIDE", "CONTRAMAL SR 100MG TAB", "TRAMADOL|100MG|TAB|MR"),        # a kind of its own
    ("CEFTAZIDIME+AVIBACTAM (SCH H1)", "ZAVICEFTA 2GM/0.5MG INJ VIAL", "AVIBACTAM+CEFTAZIDIME|0.5MG|2000MG|INJ|"),
    ("CEFUROXIME AXETIL", "SUPACEF 1.5GM INJ VIAL", "CEFUROXIME|1500MG|INJ|"),           # salt word dropped
    ("VITAMIN B COMPLEX", "BEPLEX FORTE TAB", None),                                    # no strength
    ("PARACETAMOL", "DOLO 650", None),                                                  # no form
    (None, "ERBITUX 100MG INJ VIAL", None),                                             # no molecule
])
def test_product_key(minor, desc, key):
    got = P.product_key(minor, desc)
    assert got["molecule_key"] == key
    assert set(got) == set(P.PRODUCT_COLUMNS)


def test_one_product_key_across_spellings():
    a = P.product_key("CEFUROXIME AXETIL", "SUPACEF 1.5GM INJ VIAL")
    b = P.product_key("CEFUROXIME", "BACTILEM 1.5GM INJ VIAL")
    assert a["molecule_key"] == b["molecule_key"]
    assert (a["strength"], a["form"], a["kind"]) == ("1500MG", "INJ", None)


def test_names_come_from_the_nearest_item_master():
    ims = {"2026-06": item_master([("200010", "FORMULARY", "P", "JUNE NAME 10MG TAB", "JUNE MOLECULE"),
                                   ("200020", "FORMULARY", "P", "OLD 20 5MG TAB", "MOLECULE 20")]),
           AUG: item_master([("200010", "FORMULARY", "P", "AUG NAME 10MG TAB", "  "),     # blank: next master
                             ("200020", "FORMULARY", "P", "NEW 20 5MG TAB", "MOLECULE 20B")])}
    df = frame([line(DEC, "HC05", "200010", 1, 1.0, desc="GRN 10", minor="GRN MOLECULE"),
                line(AUG, "HC05", "200010", 1, 1.0, desc="GRN 10", minor="GRN MOLECULE"),
                line(JUL, "HC05", "200020", 1, 1.0, desc="GRN 20", minor="GRN 20"),
                line(AUG, "HC05", "200030", 1, 1.0, desc="GRN 30", minor="GRN 30")])   # in no master
    got = P.with_item_master_names(df, ims)[["material_desc", "minor_group"]].values.tolist()
    assert got == [["JUNE NAME 10MG TAB", "JUNE MOLECULE"],          # Dec: June is the nearest master
                   ["AUG NAME 10MG TAB", "JUNE MOLECULE"],           # Aug: its own, a blank from June
                   ["OLD 20 5MG TAB", "MOLECULE 20"],                # Jul: Jun and Aug as near: the earlier
                   ["GRN 30", "GRN 30"]]                             # in no master: the GRN's own
    assert P.nearest_first("2026-07", ["2026-08", "2026-06", "2026-07", "2026-01"]) == \
        ["2026-07", "2026-06", "2026-08", "2026-01"]
    assert P.with_item_master_names(df, {})["material_desc"].tolist() == df["material_desc"].tolist()


# --- which prices are compared -----------------------------------------------------------------
def test_price_statuses():
    rows = [
        # 200010: bought per unit at three centres, once per strip of 30 (price and MRP 30x)
        line(AUG, "HC05", "200010", 10, 100.0, 21.0), line(AUG, "HC06", "200010", 10, 110.0, 21.0),
        line(AUG, "HC13", "200010", 10, 100.0, 21.0), line(AUG, "HC16", "200010", 2, 600.0, 630.0),
        # 200020: two lines a strip apart: the lower median keeps the per-unit one
        line(AUG, "HC05", "200020", 30, 150.0, 10.5), line(AUG, "HC06", "200020", 1, 150.0, 315.0),
        # 200030: free stock and a credit line
        line(AUG, "HC05", "200030", 5, 0.0, 10.5), line(AUG, "HC06", "200030", 1, -16.55, 165.0),
        # 101158: OPDYTA booked at Rs 0.01 beside its real price
        line(AUG, "HC05", "101158", 3, 0.03, 109450.0), line(AUG, "HC06", "101158", 1, 90000.0, 109450.0),
        # 200040: HCG's CASH&TPA MRP is 38.16 a tablet: a strip of 20 is another unit, a high
        # price at the tablet's MRP is a real price
        line(AUG, "HC05", "200040", 1, 476.2, 763.17), line(AUG, "HC06", "200040", 20, 476.2, 38.16),
        line(AUG, "HC13", "200040", 10, 1500.0, 38.16),
        # 100050: a deep discount at the same MRP is real price variation
        line(AUG, "HC05", "100050", 1, 700.0, 5319.0), line(AUG, "HC06", "100050", 1, 3100.0, 5319.0),
        line(AUG, "HC13", "100050", 1, 3100.0, 5319.0),
        # 200060: no MRP on the dear line: nothing says it is another unit
        line(AUG, "HC05", "200060", 1, 10.0, 21.0), line(AUG, "HC06", "200060", 1, 10.0, 21.0),
        line(AUG, "HC13", "200060", 1, 450.0, None),
        # 200010 again in July, per strip only: July's own median
        line(JUL, "HC05", "200010", 1, 300.0, 630.0),
    ]
    got = P.price_statuses(frame(rows), {"200040": 38.16}).tolist()
    assert got == [P.OK, P.OK, P.OK, P.OTHER_UNIT,
                   P.OK, P.OTHER_UNIT,
                   P.NO_PRICE, P.NO_PRICE,
                   P.TOKEN, P.OK,
                   P.OTHER_UNIT, P.OK, P.OK,
                   P.OK, P.OK, P.OK,
                   P.OK, P.OK, P.OK,
                   P.OK]
    assert set(P.PRICE_STATUSES) == {P.OK, P.NO_PRICE, P.TOKEN, P.OTHER_UNIT}


# --- the table and the per-product view --------------------------------------------------------
CETUXIMAB = [  # (plant, material, desc, vendor, qty, total)
    ("HC05", "102402", "LUPITUX 100MG INJ VIAL", "300100", 11, 56_100.0),       # 5,100
    ("HC05", "102402", "LUPITUX 100MG INJ VIAL", "300200", 5, 40_500.0),        # 8,100
    ("HC06", "102402", "LUPITUX 100MG INJ VIAL", "300200", 130, 1_053_000.0),   # 8,100
    ("HC06", "100177", "ERBITUX 100MG INJ VIAL", "300200", 36, 609_141.6),      # 16,920.60
    ("HC06", "100177", "ERBITUX 100MG INJ VIAL", "300200", 9, 170_442.9),       # 18,938.10
]
CETUXIMAB_QTY = sum(r[4] for r in CETUXIMAB)
CETUXIMAB_VALUE = sum(r[5] for r in CETUXIMAB)
# The usual cheapest is 8,100: the 11 vials at 5,100 are under USUAL_SHARE (10%) of the 191 bought,
# so they are bought below it (adding nothing) instead of setting it for everyone.
CETUXIMAB_USUAL = 8_100.0
CETUXIMAB_EXTRA = sum(max(0.0, total - CETUXIMAB_USUAL * qty) for *_, qty, total in CETUXIMAB)


def cetuximab_lines():
    rows = [line(AUG, pl, m, q, t, 9_365.0 if m == "102402" else 20_500.0, vendor=v, desc=d,
                 minor="CETUXIMAB") for pl, m, d, v, q, t in CETUXIMAB]
    rows += [line(AUG, "HC06", "100177", 2, 0.0, 20_500.0, vendor="300200", desc="ERBITUX 100MG INJ VIAL",
                  minor="CETUXIMAB"),                                                      # free stock
             line(AUG, "HC05", "201277", 10, 21.6, 2.92, desc="BEPLEX FORTE TAB", minor="VITAMIN B COMPLEX"),
             line(AUG, "HC05", "200278", 10, 16.9, 2.34, desc="ALDACTONE 25MG TAB", minor="SPIRONOLACTONE")]
    return rows


def test_aggregate_sums_lines_to_the_published_grain():
    t = P.aggregate(P.priced_lines(frame(cetuximab_lines()), {}, {}))
    assert list(t.columns) == P.PRICES_COLUMNS
    assert t[P.GROUP_KEYS].duplicated().sum() == 0 and t["lines"].sum() == 8
    assert t["value"].sum() == pytest.approx(CETUXIMAB_VALUE + 21.6 + 16.9)
    erb = t[(t["material"] == "100177") & (t["price_status"] == P.OK)].iloc[0]
    assert (erb["lines"], erb["qty"], erb["unit_price_min"], erb["unit_price_max"]) == (2, 45.0, 16_920.6, 18_938.1)
    assert (erb["category"], erb["segment"], erb["molecule_key"]) == ("Onco Drugs", "onco", "CETUXIMAB|100MG|INJ|")
    free = t[t["price_status"] == P.NO_PRICE].iloc[0]
    assert (free["lines"], free["qty"], free["value"]) == (1, 2.0, 0.0) and pd.isna(free["unit_price_min"])
    assert t.loc[t["material"] == "201277", "molecule_key"].isna().all()       # no strength: not compared
    assert t["lines"].dtype == "int64" and t["value"].dtype == "float64" and t["material"].dtype == "str"
    assert t["material"].tolist() == sorted(t["material"])                     # a brand's rows together


def test_products_cheapest_dearest_and_extra_paid():
    t = P.aggregate(P.priced_lines(frame(cetuximab_lines()), {}, {}))
    per = P.products(t)
    assert list(per.columns) == P.PRODUCT_VIEW_COLUMNS
    assert per["molecule_key"].tolist() == ["CETUXIMAB|100MG|INJ|", "SPIRONOLACTONE|25MG|TAB|"]
    c = per.iloc[0]
    assert (c["cheapest_price"], c["min_price"], c["max_price"], c["qty"]) == \
        (CETUXIMAB_USUAL, 5_100.0, 18_938.1, CETUXIMAB_QTY)
    # what every row paid above the usual cheapest; the 5,100 row is under it and adds nothing
    assert c["extra_paid"] == pytest.approx(CETUXIMAB_EXTRA) and c["value"] == pytest.approx(CETUXIMAB_VALUE)
    assert c["wavg_price"] == pytest.approx(CETUXIMAB_VALUE / CETUXIMAB_QTY)
    assert c["spread_ratio"] == pytest.approx(18_938.1 / CETUXIMAB_USUAL)
    assert (c["low_rows"], c["low_value"]) == (0, 0.0)          # 5,100 is not under half of 8,100
    assert (c["brands"], c["vendors"], c["plants"], c["minor_group"]) == (2, 2, 2, "CETUXIMAB")
    assert per.iloc[1]["extra_paid"] == 0.0                                     # one price: nothing extra
    assert P.products(t[t["price_status"] == P.NO_PRICE]).empty


def usual(rows, by="k"):
    return P.usual_cheapest(pd.DataFrame(rows), by, "price", carry=("who",))


def test_the_usual_cheapest_is_not_set_by_one_odd_purchase():
    # bevacizumab 400MG in Aug-26: one vial at 7,500 against 39,000 everywhere else
    got = usual([{"k": "B", "price": 7_500.0, "qty": 1, "who": "odd"},
                 {"k": "B", "price": 39_000.0, "qty": 68, "who": "a"},
                 {"k": "B", "price": 39_000.0, "qty": 45, "who": "b"}])
    assert (got.loc["B", "price"], got.loc["B", "who"]) == (39_000.0, "a")


def test_the_usual_cheapest_is_the_plain_minimum_when_enough_was_bought_at_it():
    got = usual([{"k": "A", "price": 10.0, "qty": 10, "who": "x"},       # exactly 10% of 100
                 {"k": "A", "price": 12.0, "qty": 90, "who": "y"},
                 {"k": "C", "price": 5.0, "qty": 3, "who": "z"}])        # one price: itself
    assert got["price"].to_dict() == {"A": 10.0, "C": 5.0}


def test_the_usual_cheapest_counts_up_from_the_cheapest_and_skips_empty_purchases():
    rows = [{"k": "A", "price": 1.0, "qty": 0, "who": "free"},         # no quantity: takes no part
            {"k": "A", "price": 2.0, "qty": -5, "who": "credit"},
            {"k": "A", "price": 3.0, "qty": 4, "who": "p"},            # 4%
            {"k": "A", "price": 4.0, "qty": 5, "who": "q"},            # 9%
            {"k": "A", "price": 6.0, "qty": 91, "who": "r"}]           # reaches 10% here
    assert usual(rows).loc["A", "price"] == 6.0
    assert P.usual_cheapest(pd.DataFrame(rows), "k", "price", share=0.04).loc["A", "price"] == 3.0
    two = pd.DataFrame([{"m": "2026-07", "k": "A", "price": 2.0, "qty": 1, "who": "x"},
                        {"m": "2026-08", "k": "A", "price": 9.0, "qty": 1, "who": "y"}])
    by_month = P.usual_cheapest(two, ["m", "k"], "price")["price"]
    assert by_month.to_dict() == {("2026-07", "A"): 2.0, ("2026-08", "A"): 9.0}


def test_extra_paid_and_unusually_low_rows():
    rows = pd.DataFrame({"qty": [1.0, 68.0, 10.0], "value": [7_500.0, 2_652_000.0, 400_000.0]})
    cheapest = pd.Series([39_000.0, 39_000.0, 39_000.0])
    assert P.extra_paid(rows, cheapest).tolist() == [0.0, 0.0, 10_000.0]
    assert P.unusually_low(rows, cheapest).tolist() == [True, False, False]
    assert P.extra_paid(rows, pd.Series([float("nan")] * 3)).tolist() == [0.0, 0.0, 0.0]   # no cheapest


def test_a_supplier_with_several_codes_counts_once():
    # one supplier holds one SAP code per region (Vardhman Health Specialities has 11): the
    # report counts suppliers by name, as the API does
    cet = dict(minor="CETUXIMAB", mrp=9_365.0)
    rows = [line(AUG, "HC05", "100177", 1, 8_000.0, vendor="300100", vendor_name="VARDHMAN HEALTH SPECIALITIES",
                 desc="ERBITUX 100MG INJ VIAL", **cet),
            line(AUG, "HC06", "100177", 1, 8_200.0, vendor="300101", vendor_name="Vardhman  Health Specialities",
                 desc="ERBITUX 100MG INJ VIAL", **cet),
            line(AUG, "HC06", "102402", 1, 7_000.0, vendor="300200", vendor_name="OTHER PHARMA",
                 desc="LUPITUX 100MG INJ VIAL", **cet)]
    per = P.products(P.aggregate(P.priced_lines(frame(rows), {}, {})))
    assert (per.iloc[0]["brands"], per.iloc[0]["vendors"], per.iloc[0]["plants"]) == (2, 2, 2)
    assert P.vendor_keys(pd.Series(["Vardhman  Health Specialities", "VARDHMAN HEALTH SPECIALITIES"])).nunique() == 1


def test_month_report():
    rows = cetuximab_lines() + [line(AUG, "HC05", "101158", 3, 0.03, 109450.0, desc="OPDYTA 100MG INJ VIAL",
                                     minor="NIVOLUMAB")]
    rep = P.month_report(P.aggregate(P.priced_lines(frame(rows), {}, {})))
    assert rep["lines"] == 9 and rep["by_status"][P.TOKEN] == {"lines": 1, "value": 0.03}
    assert rep["by_status"][P.NO_PRICE]["lines"] == 1 and rep["compared"]["lines"] == 6
    assert (rep["products"], rep["products_with_spread"]) == (2, 1)
    assert rep["extra_paid_vs_cheapest"] == pytest.approx(CETUXIMAB_EXTRA, abs=0.01)
    assert rep["unusually_low"] == {"rows": 0, "value": 0.0, "products": 0}
    assert rep["top_products"][0]["molecule_key"] == "CETUXIMAB|100MG|INJ|"
    assert rep["top_products"][0]["cheapest_price"] == CETUXIMAB_USUAL
    assert set(P.report_section()["rules"]["price_status"]) == set(P.PRICE_STATUSES)


def test_no_lines_give_an_empty_table():
    empty = frame([line(AUG, "HC05", "200010", 1, 1.0)]).iloc[0:0]
    t, months = P.build_table(empty, {}, {})
    assert list(t.columns) == P.PRICES_COLUMNS and t.empty and months == {}


# --- the build ----------------------------------------------------------------------------------
def test_the_build_publishes_purchase_prices(tmp_path):
    rows = cetuximab_lines() + [
        line(JUL, "HC05", "102402", 4, 32_400.0, 9_365.0, desc="LUPITUX 100MG INJ VIAL", minor="CETUXIMAB"),
        line(AUG, "BH03", "102402", 1, 8_100.0, 9_365.0, minor="CETUXIMAB")]          # Milann: out of scope
    rep = B.run(make_inputs(rows), data=tmp_path / "data", cache=tmp_path / "cache",
                builders=(("prices", B.build_prices),))
    t = pd.read_parquet(tmp_path / "data" / f"{C.PRICES}.parquet")
    assert list(t.columns) == P.PRICES_COLUMNS and sorted(t["period"].unique()) == [JUL, AUG]
    assert "BH03" not in set(t["plant"]) and t["lines"].sum() == 9
    B.check_columns(C.PRICES, t)
    v = json.loads((tmp_path / "data" / C.VALIDATION_FILE).read_text())
    assert v["months"][AUG]["prices"]["products"] == rep["months"][AUG]["prices"]["products"] == 2
    assert v["months"][JUL]["prices"]["compared"]["lines"] == 1
    assert "price_status" in v["prices"]["rules"] and v["tables"][C.PRICES]["rows"] == len(t)


def test_the_builders_run_after_hcgs_method():
    assert [s for s, _ in B.TABLE_BUILDERS] == ["oof", "prices", "generic"]


# --- the committed table ------------------------------------------------------------------------
# Figures read from the published table when the rule was chosen (Aug-26). The research that
# found the price-variation keys (Aug-26 GRN, Base Price, plain minor-group keys) put the extra
# paid above the cheapest of each molecule + strength + form at Rs 9,01,50,473: our keys and the
# price after discount gave Rs 9,00,24,529.93 (-0.14%). LIPODOX (liposomal doxorubicin filed
# under plain DOXORUBICIN) then became a product of its own instead of plain doxorubicin's
# dearest brand: Rs 8,98,82,175.13 (-0.30%) above the PLAIN minimum, which is what the research
# measured. Against the USUAL cheapest (one odd purchase no longer sets it — bevacizumab 400MG's
# single Rs 7,500 vial alone made Rs 98 L of Rs 39,000 purchases 'extra') the month's extra paid
# is Rs 5,46,42,940.44, and 40 hospital x brand x supplier rows (Rs 1,70,536) are unusually low.
DATA = Path(settings.PURCHASE_DATA_DIR)
committed = pytest.mark.skipif(not (DATA / f"{C.PRICES}.parquet").exists(),
                               reason="purchase tables not built — run python -m app.etl.purchase")
AUG_STATUS_LINES = {P.OK: 16_717, P.NO_PRICE: 272, P.TOKEN: 2, P.OTHER_UNIT: 21}
AUG_EXTRA_PAID = 54_642_940.44
AUG_EXTRA_PAID_PLAIN_MINIMUM = 89_882_175.13
RESEARCH_AUG_EXTRA_PAID = 90_150_473.45


@pytest.fixture(scope="module")
def prices_table():
    return pd.read_parquet(DATA / f"{C.PRICES}.parquet")


@committed
def test_every_scope_line_is_in_the_table(prices_table):
    report = json.loads((DATA / C.VALIDATION_FILE).read_text())
    for p, month in report["months"].items():
        t = prices_table[prices_table["period"] == p]
        scope = month["funnel"][-1]                     # the last scope step: HCG's sheet, step by step
        assert (int(t["lines"].sum()), round(float(t["value"].sum()), 2)) == (scope["lines"], scope["value"]), p
        assert month["prices"]["lines"] == scope["lines"], p


@committed
def test_august_prices(prices_table):
    if "2026-08" not in set(prices_table["period"]):
        pytest.skip("2026-08 not built")
    aug = prices_table[prices_table["period"] == "2026-08"]
    assert aug.groupby("price_status")["lines"].sum().to_dict() == AUG_STATUS_LINES
    assert (aug.loc[aug["price_status"] == P.NO_PRICE, "value"] <= 0).all()
    per = P.products(aug).set_index("molecule_key")
    cet = per.loc["CETUXIMAB|100MG|INJ|"]
    assert (cet["cheapest_price"], cet["min_price"], cet["max_price"]) == (8_100.0, 5_100.0, 18_938.1)
    bev = per.loc["BEVACIZUMAB|400MG|INJ|"]                                    # the odd Rs 7,500 vial
    assert (bev["cheapest_price"], bev["min_price"], bev["extra_paid"], bev["low_rows"]) == (39_000.0, 7_500.0, 0.0, 1)
    # liposomal doxorubicin (Rs 10,000+ a vial) is not set against plain doxorubicin (Rs 195+)
    assert per.loc["DOXORUBICIN|50MG|INJ|", "max_price"] < 1_000 < per.loc["DOXORUBICIN|50MG|INJ|LIPO", "min_price"]
    assert per["extra_paid"].sum() == pytest.approx(AUG_EXTRA_PAID, abs=1.0)
    assert (per["low_rows"].sum(), round(per["low_value"].sum(), 2)) == (40, 170_535.95)
    plain = float((per["value"] - per["min_price"] * per["qty"]).sum())       # the research's measure
    assert plain == pytest.approx(AUG_EXTRA_PAID_PLAIN_MINIMUM, abs=1.0)
    assert plain == pytest.approx(RESEARCH_AUG_EXTRA_PAID, rel=0.005)


@committed
def test_the_published_prices_are_aggregates(prices_table):
    B.check_columns(C.PRICES, prices_table)
    assert list(prices_table.columns) == P.PRICES_COLUMNS
    assert prices_table[P.GROUP_KEYS].duplicated().sum() == 0 and prices_table["lines"].min() >= 1
    assert set(prices_table["price_status"]) <= set(P.PRICE_STATUSES)
