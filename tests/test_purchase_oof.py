# tests/test_purchase_oof.py — HCG's out-of-formulary purchase method on tiny made-up data.
#
# The description parser is checked on real HCG product descriptions and Item Master minor
# groups (product names only). The method — HCG's per-line formulas, the suggestion order
# (HCG's sheet, HCG's pairs of other months, HCG's master, our rule, none), parked and
# unpriced lines, rule PC prices, the aggregation and the golden guard — runs on frames and
# objects built here, through the same build.run() the real build uses: no workbook, no
# app/data, no cache is touched (every directory is a tmp_path).
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.etl.formulary.discover import BuildError
from app.etl.purchase import build as B
from app.etl.purchase import config as C
from app.etl.purchase import grn as G
from app.etl.purchase import molecule as Mo
from app.etl.purchase import oof as O
from app.etl.purchase.masters import (PAIR_COLUMNS, PARKED_COLUMNS, SUMMARY_COLUMNS, HcgWorkbook,
                                      MainSheet, OofMaster, OofMasterRow)
from app.etl.spend import oof as S
from app.etl.spend.masters import BrandList, PricedBrand


# --- the description parser (real descriptions) ------------------------------------------
@pytest.mark.parametrize("minor_group, key", [
    ("ACECLOFENAC+TRAMADOL HCL (SCH H1)", "ACECLOFENAC+TRAMADOL"),    # schedule tag, salt
    ("BALOFLOXACIN(SCH H1)", "BALOFLOXACIN"),
    ("MONTELUKAST SODIUM", "MONTELUKAST"),
    ("CALCIUM CARBONATE+VITAMIN D3", "CALCIUMCARBONATE+CHOLECALCIFEROL"),   # all salt words: kept
    ("AMOXYCILLIN+CLAVULANIC ACID", "AMOXICILLIN+CLAVULANICACID"),      # synonym
    ("LIGNOCAINE HCL", "LIDOCAINE"),
    ("N-ACETYLCYSTEINE", "ACETYLCYSTEINE"),
    ("AMPHOTERICIN\xa0B (LIPOSOMAL)", "AMPHOTERICINB"),                # bracketed text dropped
    ("PARACETAMOL / CAFFEINE", "CAFFEINE+PARACETAMOL"),                  # parts sorted
    ("(SCH H1)", None), ("  ", None), (None, None),
])
def test_molecule_key(minor_group, key):
    assert Mo.molecule_key(minor_group) == key


def test_one_molecule_two_names():
    assert Mo.molecule_key("NORMAL SALINE") == Mo.molecule_key("SODIUM CHLORIDE")
    assert Mo.molecule_key("ACETAMINOPHEN+TRAMADOL HCL (SCH H1)") == \
        Mo.molecule_key("TRAMADOL HYDROCHLORIDE + PARACETAMOL")
    assert Mo.molecule_parts("A+B") == {"A", "B"} and Mo.molecule_parts(None) == frozenset()


def test_close_molecules_needs_the_same_number_of_parts():
    known = ["NICORANDIL", "NIFEDIPINE", "NICORANDIL+ASPIRIN"]
    assert Mo.close_molecules("NICRONDIL", known) == ["NICORANDIL"]
    assert Mo.close_molecules("NICRONDIL+ASPIRIN", known) == ["NICORANDIL+ASPIRIN"]
    assert Mo.close_molecules("PARACETAMOL", known) == [] and Mo.close_molecules(None, known) == []


@pytest.mark.parametrize("desc, clean", [
    ("DIRIPLEVO 25MGINJ VIAL", "DIRIPLEVO 25MG INJ VIAL"),              # unit glued to a word
    ("ADTRINIB 200MGTAB 30'S CONTAINER", "ADTRINIB 200MG TAB 30'S CONTAINER"),
    ("AM AMINO 10%INJ 200ML BOTTLE", "AM AMINO 10% INJ 200ML BOTTLE"),
    ("ADVITAM WHEY PROTEIN POW 200GM CONTAINER", "ADVITAM WHEY PROTEIN POWDER 200GM CONTAINER"),
    ("ARGIPREG GRANULES 6.5GM SACHET", "ARGIPREG POWDER 6.5GM SACHET"),
    ("ATOCOR CV 10/75MG CAP", "ATOCOR CV 10MG/75MG CAP"),               # 10/75MG = 10MG + 75MG
    ("EMBLAVEO 1.5/0.5GM INJ VIAL", "EMBLAVEO 1.5GM/0.5GM INJ VIAL"),
])
def test_clean_desc(desc, clean):
    assert Mo.clean_desc(desc) == clean


@pytest.mark.parametrize("desc, strength, dose", [
    ("COPTRADOL 50MG/ML INJ 2ML AMP", "100MG", "100MG"),               # per mL x the one volume
    ("TERLITAS 1MG/ML INJ 10ML VIAL", "10MG", "10MG"),
    ("MUCOMIX 200MG/1ML INJ 1ML VIAL", "200MG", "200MG"),
    ("PAGENAX 120MG/ML INJ", "120MG", "120MG"),                         # no volume: the dose
    ("ERYKINE 40000IU/1ML INJ PFS", "40000IU", "40000IU"),
    ("ATOCOR CV 10/75MG CAP", "10MG|75MG", "10MG|75MG"),
    ("DIRIPLEVO 25MGINJ VIAL", "25MG", "25MG"),
    ("AM AMINO 10%INJ 500ML BOTTLE", "10%|500ML", "10%"),               # dose alone: no volume
    ("ADVITAM WHEY PROTEIN POW 200GM CONTAINER", "200000MG", ""),      # 200 GM: a pack weight
    ("ARGIPREG GRANULES 6.5GM SACHET", "6500MG", "6500MG"),
])
def test_strength_and_dose(desc, strength, dose):
    assert (Mo.strength_key(desc), Mo.dose_key(desc)) == (strength, dose)


def test_the_fixes_change_what_the_shared_parser_reads():
    assert S.strength_key("DIRIPLEVO 25MGINJ VIAL") == "" and Mo.strength_key("DIRIPLEVO 25MGINJ VIAL") == "25MG"
    assert S.form_key("ADVITAM WHEY PROTEIN POW 200GM CONTAINER") == ""
    assert Mo.form_key("ADVITAM WHEY PROTEIN POW 200GM CONTAINER") == "POWDER"


@pytest.mark.parametrize("desc, pack", [
    ("AM AMINO 10%INJ 500ML BOTTLE", 500.0),
    ("GLYCINE 1.5% INJ 3000ML PB-PENTAGON", 3000.0),
    ("ADVITAM WHEY PROTEIN POW 200GM CONTAINER", 200_000.0),           # mg, like a dose
    ("BISOTRIL 2.5MG TAB", None),
])
def test_pack_size(desc, pack):
    assert Mo.pack_size(desc) == pack


def test_form_groups_and_kinds():
    assert [Mo.form_group(Mo.form_key(d)) for d in (
        "ATOCOR CV 10/75MG CAP", "BISOTRIL 2.5MG TAB", "LACTIHEP PLUS SYR", "OTOGESIC DROPS",
        "COPTRADOL 50MG/ML INJ 2ML AMP", "PLAIN PRODUCT")] == \
        ["ORAL SOLID", "ORAL SOLID", "LIQUID", "LIQUID", "INJ", ""]
    assert Mo.kind_key("CONTRAMAL SR 100MG TAB") == "MR"
    assert Mo.kind_key("BETAONE XL 25MG TAB") == "MR"
    assert Mo.kind_key("ACIVIR-DT 400MG TAB") == "MD"                    # mouth-dissolving
    assert Mo.kind_key("CLONAFIT-MD 0.25MG TAB") == "MD"
    assert Mo.kind_key("PACLIALL 100MG INJ VIAL", "PACLITAXEL PROTEIN BOUND") == "NAB"
    # liposomal doxorubicin filed under plain DOXORUBICIN is not the plain product
    assert Mo.kind_key("LIPODOX 50MG INJ VIAL", "DOXORUBICIN") == "LIPO"
    assert Mo.kind_key("ADRIM 50MG INJ VIAL", "DOXORUBICIN") == ""
    assert Mo.kind_key("BISOTRIL 2.5MG TAB", "BISOPROLOL FUMARATE") == ""


def test_descriptor_words_name_the_variant():
    assert Mo.descriptor_words("FERTIRO XT TAB") == {"XT"}
    assert Mo.descriptor_words("ECOSPRIN-AV 75/10MG CAP") == {"AV"}
    assert Mo.descriptor_words("LIVOGEN-XT TAB 10'S") == {"XT"}
    assert Mo.descriptor_words(None) == frozenset()


def test_describe():
    p = Mo.describe("TRAMADOL HCL (SCH H1)", "COPTRADOL 50MG/ML INJ 2ML AMP")
    assert p == Mo.Product(molecule="TRAMADOL", strength="100MG", dose="100MG", form="INJ",
                           form_group="INJ", kind="", words=frozenset(), pack=2.0)


# --- made-up data ---------------------------------------------------------------------------
def grn_line(period, plant, material, qty, base, unit_mrp, tax=5.0, *, vendor="300100",
             vendor_name="MEDS DISTRIBUTOR", mvt="101", sloc="S124", desc=None, minor="MOLECULE"):
    """One GRN line as grn.read_grn_files gives it (no identifiers exist there)."""
    return {"period": period, "plant": plant, "sloc": sloc, "material": material,
            "material_desc": desc or f"ITEM {material}", "vendor_code": vendor,
            "vendor_name": vendor_name, "mvt_type": mvt, "gr_date": pd.Timestamp(f"{period}-04"),
            "gr_qty": qty, "uom": "EA", "base_price": base, "discount_value": 0.0,
            "net_price": base, "total_wo_tax": base * qty, "tax_pct": tax, "unit_mrp": unit_mrp,
            "manufacturer": "GRN MAKER", "material_type": "ZNOC", "minor_category": "DRUG",
            "major_group": "DRUGS", "minor_group": minor, "grn_formulary": "FORMULARY"}


def item_master(rows) -> pd.DataFrame:
    """(material, flag, desc, minor group) -> masters.item_master_frame rows."""
    recs = [{"material": m, "formulary_class": "FORMULARYP" if f == "FORMULARY" else f,
             "formulary_flag": f, "tier": "P" if f == "FORMULARY" else None, "material_desc": d,
             "minor_group": g, "manufacturer": "MAKER"} for m, f, d, g in rows]
    return pd.DataFrame(recs)


def pairs(period, rows) -> pd.DataFrame:
    """(plant, material, base, suggested, price, mrp, gst, lines) -> HCG's pairs."""
    recs = [{"period": period, "plant": pl, "material": m, "base_price": b, "material_desc": f"ITEM {m}",
             "suggested": s, "suggested_desc": f"SUG {s}" if s else None, "sug_price": pr,
             "sug_mrp": mrp, "sug_gst": gst, "lines": n, "qty": float(n)}
            for pl, m, b, s, pr, mrp, gst, n in rows]
    return pd.DataFrame(recs, columns=PAIR_COLUMNS)


def workbook(period, pair_rows, parked=(), summary=()) -> HcgWorkbook:
    pk = pd.DataFrame([{"period": period, "plant": pl, "material": m, "material_desc": f"ITEM {m}",
                        "base_price": b, "lines": n, "qty": float(n)} for pl, m, b, n in parked],
                      columns=PARKED_COLUMNS)
    sm = pd.DataFrame([{"period": period, "plant": pl, "centre_name": name, "used_pp": c, "used_mrp": d,
                        "used_margin": d - c, "sug_pp": h, "sug_mrp": i, "sug_margin": i - h,
                        "loss": (i - h) - (d - c)} for pl, name, c, d, h, i in summary],
                      columns=SUMMARY_COLUMNS)
    pr = pairs(period, pair_rows)
    return HcgWorkbook(name=f"OOF {period}.xlsx", period=period, ws_sheet="WS", ws_lines=int(pr["lines"].sum()),
                       ws_unpriced=int(pr["suggested"].isna().sum()), pairs=pr, parked_sheet=None,
                       parked=pk, summary_sheet="Summary", summary=sm, summary_computed=0, titles=[],
                       warnings=[])


def master(rows) -> OofMaster:
    """(row, OOF material, suggested, price, mrp, gst) -> HCG's OOF master."""
    m = OofMaster(name="master-2026-27.xlsb", sheet="Sheet1", fy=(2026, 2027), rows=len(rows))
    for n, mat, sug, price, mrp, gst in rows:
        m.entries[mat] = OofMasterRow(row=n, material=mat, material_desc=f"ITEM {mat}", generic_name=None,
                                      suggested=sug, suggested_desc=f"MASTER {sug}", price=price, mrp=mrp, gst=gst)
    return m


def main_sheet(brands) -> MainSheet:
    """(material, desc, block, p_rank, price, mrp) -> the CASH&TPA price list and blocks."""
    blocks = pd.DataFrame([{"material": m, "material_desc": d, "block_id": b, "p_rank": r}
                           for m, d, b, r, _, _ in brands])
    prices = {m: PricedBrand(material=m, material_desc=d, price=p, mrp=mrp, gst=5.0, category="P")
              for m, d, _, _, p, mrp in brands if p is not None}
    cash = BrandList(path=Path("main.xlsx"), sheet="CASH&TPA", fy=(2026, 2027))
    return MainSheet(path=Path("main.xlsx"), cash=cash, prices=prices, blocks=blocks)


def priced(price, mrp, gst, src):
    return dict(price=price, mrp=mrp, gst=gst, price_source=src, status=O.COUNTED)


# --- HCG's formulas --------------------------------------------------------------------------
def test_line_values_follow_hcgs_worksheet():
    lines = pd.DataFrame([grn_line("2026-08", "HC05", "200010", 10.0, 100.0, 210.0),
                          grn_line("2026-08", "HC05", "200020", 4.0, 10.0, 21.0),
                          grn_line("2026-08", "HC05", "200030", 3.0, 7.0, 10.5)])
    sug = [O.Suggestion(source=O.HCG_SHEET, material="210010", **priced(50.0, 105.0, 5.0, O.PRICE_HCG_SHEET)),
           O.Suggestion(source=O.HCG_SHEET, material="210020", **priced(2.0, 50.0, 0.0, O.PRICE_HCG_SHEET)),
           O.NO_SUGGESTION]
    v = O.with_suggestions(lines, sug)
    assert set(O.VALUE_COLUMNS) <= set(v.columns)
    # abb MRP = 210 / 105 x 100 = 200; PP 1000; MRP 2000; suggested abb 100: PP 500, MRP 1000
    first = v.iloc[0][O.VALUE_COLUMNS].tolist()
    assert first == pytest.approx([200.0, 1000.0, 2000.0, 1000.0, 500.0, 1000.0, 500.0, -500.0])
    # HCG's 0% GST typed as is: abb = MRP; margin (50 - 2) x 4 = 192 against (20 - 10) x 4 = 40
    assert v.iloc[1][["actual_margin", "suggested_margin", "margin_loss"]].tolist() == \
        pytest.approx([40.0, 192.0, 152.0])
    # no alternative: purchase value kept, nothing in the margins or the loss
    assert v.iloc[2][["total_pp", "total_mrp"]].tolist() == pytest.approx([21.0, 30.0])
    assert v.iloc[2][["actual_margin", "sug_total_pp", "sug_total_mrp", "suggested_margin",
                      "margin_loss"]].tolist() == [0.0] * 5
    t = O.totals(v)
    assert (t["margin_loss"], t["gain_loss"], t["better_loss"]) == (-348.0, 152.0, -500.0)
    assert t["loss_pct"] == pytest.approx(-348.0 / 692.0 * 100, abs=0.01)
    with pytest.raises(ValueError, match="one suggestion per line"):
        O.with_suggestions(lines, sug[:2])


# --- source 1: HCG's worksheet of the month ---------------------------------------------------
def test_hcgs_sheet_decides_a_month_it_covers():
    wb = workbook("2026-07", [
        ("HC05", "200010", 100.0, "210010", 50.0, 105.0, 5.0, 3),
        ("HC05", "200020", 10.0, "210020", 2.0, 50.0, 0.0, 1),
        ("HC05", "200050", 5.0, "210051", 3.0, 10.5, 5.0, 1),          # two suggestions for one key:
        ("HC05", "200050", 6.0, "210052", 4.0, 10.5, 5.0, 2),          # the base price decides
        ("HC06", "200050", 5.0, "210051", 3.0, 10.5, 5.0, 1),
        ("HC06", "200050", 6.0, "210052", 4.0, 10.5, 5.0, 3),
        ("HC06", "200060", 8.0, None, None, None, None, 1),            # HCG's own #N/A
    ], parked=[("HC05", "200030", 7.0, 1)])
    lines = pd.DataFrame([
        grn_line("2026-07", "HC05", "200010", 1.0, 100.0, 210.0),
        grn_line("2026-07", "HC05", "200010", 2.0, 101.0, 210.0),      # another price, one suggestion
        grn_line("2026-07", "HC05", "200030", 1.0, 7.0, 10.5),         # parked
        grn_line("2026-07", "HC05", "200030", 1.0, 8.0, 10.5),         # same brand, not parked: none
        grn_line("2026-07", "HC05", "200040", 1.0, 9.0, 10.5),         # not on HCG's sheet: none
        grn_line("2026-07", "HC05", "200050", 1.0, 6.0, 10.5),
        grn_line("2026-07", "HC06", "200050", 1.0, 9.0, 10.5),         # no base price: most lines
        grn_line("2026-07", "HC06", "200060", 1.0, 8.0, 10.5),
    ])
    got = O.sheet_suggestions(lines, wb)
    assert [(s.source, s.status, s.material) for s in got] == [
        (O.HCG_SHEET, O.COUNTED, "210010"), (O.HCG_SHEET, O.COUNTED, "210010"),
        (O.HCG_SHEET, O.PARKED, None), (O.NONE, O.NO_ALTERNATIVE, None),
        (O.NONE, O.NO_ALTERNATIVE, None), (O.HCG_SHEET, O.COUNTED, "210052"),
        (O.HCG_SHEET, O.COUNTED, "210052"), (O.NONE, O.NO_ALTERNATIVE, None)]
    assert (got[0].price, got[0].mrp, got[0].gst, got[0].price_source, got[0].desc) == \
        (50.0, 105.0, 5.0, O.PRICE_HCG_SHEET, "SUG 210010")


# --- sources 2-5 ------------------------------------------------------------------------------
def test_pair_months_prefer_the_nearest_earlier_sheet():
    sheets = ["2026-06", "2026-07", "2026-08", "2026-10"]
    assert O.pair_months("2026-09", sheets) == ["2026-08", "2026-07", "2026-06", "2026-10"]
    assert O.pair_months("2026-09", sheets, earlier_only=True) == ["2026-08", "2026-07", "2026-06"]
    assert O.pair_months("2026-06", sheets, earlier_only=True) == []
    assert O.pair_months("2026-07", sheets) == ["2026-06", "2026-08", "2026-10"]


def test_sheet_pair_book_takes_the_suggestion_on_most_lines():
    wb = workbook("2026-07", [("HC05", "200010", 1.0, "210010", 5.0, 9.0, 5.0, 1),
                              ("HC06", "200010", 1.0, "210011", 5.0, 9.0, 5.0, 3),
                              ("HC06", "200020", 1.0, None, None, None, None, 4)])
    book = O.sheet_pair_book(wb)
    assert book == {"200010": O.SheetPair("2026-07", "210011", "SUG 210011")}


def test_rule_pc_prices():
    m = master([(3, "200060", "210010", 48.0, 105.0, 5.0), (4, "200070", "210010", 45.0, 100.0, 5.0)])
    pb = O.PriceBook(m, main_sheet([("210010", "X", 1, 1, 60.0, 120.0), ("210020", "Y", 2, 1, 7.0, 21.0)]).prices)
    assert pb.for_brand("210010") == O.Price(45.0, 100.0, 5.0, O.PRICE_LIST_LOWEST)      # lowest master row
    assert pb.for_brand("210020") == O.Price(7.0, 21.0, 5.0, O.PRICE_CASH_TPA)
    assert pb.for_brand("210099") is None
    assert O.PriceBook.master_row(m.entries["200060"]) == O.Price(48.0, 105.0, 5.0, O.PRICE_LIST_ROW)


# Pool: CASH&TPA brands, FORMULARY in the month, with a price (description, block, P-rank).
POOL_BRANDS = [
    ("300001", "FORMAMOX 625MG TAB", 1, 2, 10.0, 30.0),
    ("300002", "BESTAMOX 625MG TAB", 1, 1, 12.0, 30.0),           # P1 of the block
    ("300003", "CAPAMOX 625MG CAP", 2, 1, 9.0, 30.0),
    ("300004", "NOPRICE 625MG TAB", 1, 3, None, None),            # no FY price: not in the pool
    ("300005", "NS 0.9% 100ML PB", 3, 1, 20.0, 40.0),
    ("300006", "PROTEIN POW 400GM CONTAINER", 4, 1, 300.0, 700.0),
    ("300007", "BIGPRO POW 1000GM CONTAINER", 4, 1, 500.0, 1000.0),   # pack 5x: too big
    ("300008", "TRAMACET 37.5/325MG TAB", 5, 1, 5.0, 15.0),
    ("300009", "NICOFORM 5MG TAB", 6, 1, 4.0, 10.0),
    ("300010", "NOTFORMULARY 625MG TAB", 1, 1, 1.0, 30.0),        # NON FORMULARY this month
    ("300011", "EMESET 4MG MD TAB", 7, 2, 3.0, 9.0),
    ("300012", "VOMIKIND 4MG TAB", 7, 1, 2.0, 9.0),
]
POOL_MINOR = {"300001": "AMOXICILLIN+CLAVULANIC ACID", "300002": "AMOXYCILLIN + CLAVULANIC ACID",
              "300003": "AMOXICILLIN+CLAVULANIC ACID", "300004": "AMOXICILLIN+CLAVULANIC ACID",
              "300005": "SODIUM CHLORIDE", "300006": "WHEY PROTEIN", "300007": "WHEY PROTEIN",
              "300008": "PARACETAMOL+TRAMADOL HCL (SCH H1)", "300009": "NICORANDIL",
              "300010": "AMOXICILLIN+CLAVULANIC ACID", "300011": "ONDANSETRON",
              "300012": "ONDANSETRON"}


@pytest.fixture()
def pool():
    ms = main_sheet(POOL_BRANDS)
    formulary = set(POOL_MINOR) - {"300010"}
    return O.FormularyPool.from_main_sheet(ms.blocks, ms.prices, formulary, POOL_MINOR)


@pytest.mark.parametrize("minor, desc, brand, level", [
    ("AMOXYCILLIN+CLAVULANIC ACID", "OOFAMOX 625MG TAB", "300002", "A"),    # P1 of the block
    ("AMOXYCILLIN+CLAVULANIC ACID", "OOFAMOX 625MG CAP", "300003", "A"),    # same form first
    ("AMOXICILLIN+CLAVULANIC ACID", "OOFAMOX 625MG SYP", None, None),       # no liquid in the pool
    ("NORMAL SALINE", "OOF NS 0.9% 500ML PB", "300005", "C"),               # same dose, other pack
    ("WHEY PROTEIN", "OOFPRO POW 200GM CONTAINER", "300006", "D"),          # no dose, pack <= 4x
    ("TRAMADOL HCL", "OOFTRAM 50MG TAB", None, None),                       # no tramadol 50MG brand
    ("PARACETAMOL+TRAMADOL+DOMPERIDONE", "OOFTRIO 37.5/325MG TAB", "300008", "F"),   # combination
    ("NICRONDIL", "OOFNIC 5MG TAB", "300009", "A~"),                        # molecule misspelt
    ("ONDANSETRON", "OOFSET 4MG MD TAB", "300011", "A"),                    # MD kind before P1
    ("ONDANSETRON", "OOFSET 4MG TAB", "300012", "A"),
    (None, "NO MOLECULE 5MG TAB", None, None),
])
def test_the_rule_levels(pool, minor, desc, brand, level):
    m = pool.match(Mo.describe(minor, desc))
    if brand is None:
        assert m is None
    else:
        assert (m.brand.material, m.level) == (brand, level)


def test_the_pool_holds_priced_formulary_brands_with_a_molecule(pool):
    assert {b.material for b in pool.brands} == set(POOL_MINOR) - {"300004", "300010"}
    b = {x.material: x for x in pool.brands}
    assert (b["300002"].block_id, b["300002"].position, b["300002"].p_rank) == (1, 1, 1)
    assert O.level_matches("B", Mo.describe("X", "A 5MG TAB"), Mo.describe("X", "B 5MG CAP"))
    with pytest.raises(ValueError, match="unknown rule level"):
        O.level_matches("Z", Mo.describe("X", "A"), Mo.describe("X", "B"))


def test_the_chain_order(pool):
    books = [{"200010": O.SheetPair("2026-08", "210010", "SUG 210010"),
              "200020": O.SheetPair("2026-08", "210099", "SUG 210099")},
             {"200010": O.SheetPair("2026-06", "210011", "OLDER")}]
    m = master([(3, "200010", "210050", 1.0, 2.0, 5.0), (4, "200060", "210010", 48.0, 105.0, 5.0),
                (5, "200070", "210010", 45.0, 100.0, 5.0)])
    ms = main_sheet(POOL_BRANDS)
    chain = O.Chain(books, m, pool, O.PriceBook(m, ms.prices))
    amox = Mo.describe("AMOXICILLIN+CLAVULANIC ACID", "OOFAMOX 625MG TAB")
    got = {k: chain.suggest(k, amox) for k in ("200010", "200020", "200060", "200080")}
    # 1: the nearest sheet's pair, priced at the master's lowest price of that brand
    assert (got["200010"].source, got["200010"].material, got["200010"].price,
            got["200010"].price_source) == (O.HCG_SHEET_PAIR, "210010", 45.0, O.PRICE_LIST_LOWEST)
    # 2: a pair no price list prices: shown, not counted
    assert (got["200020"].source, got["200020"].status) == (O.HCG_SHEET_PAIR, O.UNPRICED)
    # 3: the master row's own price, not the lowest
    assert (got["200060"].source, got["200060"].price, got["200060"].price_source) == \
        (O.HCG_LIST, 48.0, O.PRICE_LIST_ROW)
    # 4: our rule, CASH&TPA price
    assert (got["200080"].source, got["200080"].material, got["200080"].level, got["200080"].price,
            got["200080"].price_source) == (O.FORMULARY_RULE, "300002", "A", 12.0, O.PRICE_CASH_TPA)
    # 5: nothing
    assert chain.suggest("200090", Mo.describe("UNKNOWN", "OOF 1MG TAB")) == O.NO_SUGGESTION


# --- aggregation -------------------------------------------------------------------------------
def test_aggregate_sums_lines_to_the_published_grain():
    lines = pd.DataFrame([
        grn_line("2026-08", "HC05", "200010", 10.0, 100.0, 210.0),
        grn_line("2026-08", "HC05", "200010", 5.0, 90.0, 210.0),            # same group, other price
        grn_line("2026-08", "HC05", "200010", 1.0, 100.0, 210.0, vendor="300200", vendor_name="OTHER"),
        grn_line("2026-08", "HC05", "100020", 2.0, 100.0, 105.0),
        grn_line("2026-08", "HC05", "200030", 3.0, 7.0, 10.5),
    ])
    lines["category"] = lines["material"].map(lambda m: "Onco Drugs" if m[0] == "1" else "Other Drugs")
    lines["segment"] = lines["material"].map(lambda m: "onco" if m[0] == "1" else "non-onco")
    s10 = O.Suggestion(source=O.HCG_SHEET, material="210010", **priced(50.0, 105.0, 5.0, O.PRICE_HCG_SHEET))
    s20 = O.Suggestion(source=O.HCG_SHEET, material="110020", **priced(60.0, 84.0, 0.0, O.PRICE_HCG_SHEET))
    v = O.with_suggestions(lines, [s10, s10, s10, s20, O.PARKED_LINE])
    t = O.aggregate(v)
    assert list(t.columns) == O.OOF_COLUMNS and len(t) == 4
    assert all(str(t[c].dtype) == "str" for c in O.TEXT_COLUMNS) and t["lines"].dtype == "int64"
    assert (t["lines"].sum(), t["qty"].sum()) == (len(v), v["gr_qty"].sum())
    for c in ("total_pp", "total_wo_tax", "total_mrp", "actual_margin", "sug_total_pp",
              "sug_total_mrp", "suggested_margin", "margin_loss"):
        assert t[c].sum() == pytest.approx(v[c].sum()), c
    assert (t["gain_loss"] + t["better_loss"]).sum() == pytest.approx(t["margin_loss"].sum())
    kr = t[(t["material"] == "200010") & (t["vendor_code"] == "300100")].iloc[0]
    # two lines: 10 x (200 - 100) + 5 x (200 - 90) = 1550 earned; (100 - 50) x 15 = 750 possible
    assert (kr["lines"], kr["qty"], kr["base_price_min"], kr["base_price_max"]) == (2, 15.0, 90.0, 100.0)
    assert (kr["actual_margin"], kr["suggested_margin"], kr["margin_loss"]) == (1550.0, 750.0, -800.0)
    assert (kr["gain_loss"], kr["better_loss"], kr["segment"], kr["category"]) == \
        (0.0, -800.0, "non-onco", "Other Drugs")
    onco = t[t["material"] == "100020"].iloc[0]
    assert (onco["margin_loss"], onco["gain_loss"], onco["segment"]) == (48.0, 48.0, "onco")
    park = t[t["status"] == O.PARKED].iloc[0]
    assert (park["suggestion_source"], park["total_pp"], park["margin_loss"]) == (O.HCG_SHEET, 21.0, 0.0)
    assert pd.isna(park["suggested_material"]) and pd.isna(park["rule_level"])
    c = O.centre_figures(t)
    assert list(c.columns) == ["period", "plant"] + O.CENTRE_COLUMNS
    # counted rows only (the parked line is out): PP 1450 + 100 + 200, MRP 3000 + 200 + 200
    assert c.iloc[0][O.CENTRE_COLUMNS].tolist() == pytest.approx(
        [1750.0, 3400.0, 1650.0, 920.0, 1768.0, 848.0, -802.0])


# --- HCG's emailed figures --------------------------------------------------------------------
def test_email_figures(tmp_path):
    assert O.load_email_figures(tmp_path) == {}
    (tmp_path / O.EMAIL_FILE).write_text(
        "period,plant,actual_margin,suggested_margin,margin_loss,source\n"
        "2026-08, hc05 ,492782,1816364,1323583,HCG email\n")
    e = O.load_email_figures(tmp_path)
    assert e == {("2026-08", "HC05"): O.EmailFigure("2026-08", "HC05", 492782.0, 1816364.0, 1323583.0,
                                                    "HCG email")}
    for body, error in [("2026-8,HC05,1,2,3,x\n", "needs a period"),
                        ("2026-08,HC05,1,x,3,x\n", "must be numbers"),
                        ("2026-08,HC05,1,2,3,x\n2026-08,HC05,1,2,3,y\n", "listed twice")]:
        (tmp_path / O.EMAIL_FILE).write_text("period,plant,actual_margin,suggested_margin,margin_loss,"
                                             "source\n" + body)
        with pytest.raises(BuildError, match=error):
            O.load_email_figures(tmp_path)


# --- the build, end to end on made-up inputs --------------------------------------------------
SCOPE = C.Scope(mvt_keep=frozenset({"101"}), series_keep=frozenset({"1", "2"}), sloc=frozenset({"S125"}),
                plant_prefix=("BH",), plant=frozenset({"HM03"}))
AUG, SEP = "2026-08", "2026-09"


def grn_lines():
    return pd.DataFrame([
        # Aug, HCG's worksheet: KR loses 152 on 200020, gains 500 on 200010; HC06 loses 48
        grn_line(AUG, "HC05", "200010", 10.0, 100.0, 210.0),
        grn_line(AUG, "HC05", "200020", 4.0, 10.0, 21.0),
        grn_line(AUG, "HC05", "200030", 1.0, 7.0, 10.5),                   # parked by HCG
        grn_line(AUG, "HC05", "200040", 1.0, 9.0, 10.5),                   # HCG left it out
        grn_line(AUG, "HC06", "100050", 2.0, 100.0, 105.0),
        grn_line(AUG, "HC06", "200099", 2.0, 100.0, 105.0),                # FORMULARY: not OOF
        grn_line(AUG, "HC05", "200010", -1.0, 100.0, 210.0, mvt="122"),    # a return
        grn_line(AUG, "HC05", "200010", 1.0, 100.0, 210.0, sloc="S125"),   # out of scope
        grn_line(AUG, "BH03", "200010", 1.0, 100.0, 210.0),                # Milann
        # Sep, no HCG worksheet: sources 2-5
        grn_line(SEP, "HC05", "200010", 1.0, 100.0, 210.0),                # Aug's pair, master's lowest price
        grn_line(SEP, "HC05", "200060", 1.0, 100.0, 210.0),                # HCG's master
        grn_line(SEP, "HC05", "200080", 1.0, 20.0, 31.5, desc="OOFAMOX 625MG TAB"),   # our rule
        grn_line(SEP, "HC05", "200020", 1.0, 10.0, 21.0),                  # Aug's pair, no price
        grn_line(SEP, "HC05", "200090", 1.0, 10.0, 21.0),                  # nothing
    ])


AUG_SHEET = [("HC05", "200010", 100.0, "210010", 50.0, 105.0, 5.0, 1),
             ("HC05", "200020", 10.0, "210020", 2.0, 50.0, 0.0, 1),
             ("HC06", "100050", 100.0, "110050", 60.0, 84.0, 0.0, 1)]
# HCG's Summary: used PP / MRP and suggested PP / MRP per centre (KR loss -348, HC06 48)
AUG_SUMMARY = [("HC05", "KR", 1040.0, 2080.0, 508.0, 1200.0), ("HC06", "DR", 200.0, 200.0, 120.0, 168.0)]


def make_inputs(summary=AUG_SUMMARY, email=(1040.0, 692.0, -348.0), periods=(AUG, SEP), sheet=True,
                without_sheet=()) -> B.Inputs:
    frame = grn_lines()
    frame = frame[frame["period"].isin(periods)].reset_index(drop=True)
    lines, funnel = G.apply_scope(frame, SCOPE)
    oof = ["200010", "200020", "200030", "200040", "100050", "200060", "200080", "200090"]
    im_rows = [(m, "OUT OF FORMULARY", f"ITEM {m}", "MOLECULE " + m) for m in oof]
    im_rows[6] = ("200080", "OUT OF FORMULARY", "OOFAMOX 625MG TAB", "AMOXYCILLIN+CLAVULANIC ACID")
    im_rows += [("200099", "FORMULARY", "ITEM 200099", "X")] + \
        [(m, "FORMULARY", None, POOL_MINOR[m]) for m in POOL_MINOR if m != "300010"]
    ims = {AUG: item_master(im_rows), SEP: item_master(im_rows)}
    files = [G.GrnFileReport(file=f"GRN {p}.xlsx", sheet="GRN", period=p, rows=int((frame["period"] == p).sum()),
                             lines=int((frame["period"] == p).sum()), blank_rows=0, blank_material={},
                             mvt={}, returns=0, dates={"date": 1}, value=0.0, qty=0.0, source_size=1,
                             source_mtime_ns=1) for p in sorted(periods)]
    hcg = {AUG: workbook(AUG, AUG_SHEET, parked=[("HC05", "200030", 7.0, 1)], summary=summary)} if sheet else {}
    m = master([(3, "200060", "210010", 48.0, 105.0, 5.0), (4, "200070", "210010", 45.0, 100.0, 5.0)])
    email_rows = {(AUG, "HC05"): O.EmailFigure(AUG, "HC05", *email, "test")} if email else {}
    return B.Inputs(scope=SCOPE, grn=G.GrnRead(folder=None, frame=frame, files=files), lines=lines,
                    funnel=funnel, periods=sorted(periods), item_masters=ims,
                    item_master_files={AUG: "IM Aug.xlsx", SEP: "IM Sep.xlsx"}, hcg=hcg, master=m,
                    main=main_sheet(POOL_BRANDS), email=email_rows,
                    without_hcg_sheet=frozenset(without_sheet), log=lambda *_: None)


def read_table(data, name=C.OOF):
    return pd.read_parquet(Path(data) / f"{name}.parquet")


def test_the_build_publishes_hcgs_method(tmp_path):
    rep = B.run(make_inputs(), data=tmp_path / "data", cache=tmp_path / "cache")
    t = read_table(tmp_path / "data")
    aug, sep = t[t["period"] == AUG], t[t["period"] == SEP]
    # Aug: HCG's sheet only; parked and left-out lines shown, not counted
    assert aug.groupby("status")["lines"].sum().to_dict() == {"counted": 3, "no_alternative": 1, "parked": 1}
    assert set(aug["suggestion_source"]) == {O.HCG_SHEET, O.NONE}
    assert aug.groupby("plant")["margin_loss"].sum().to_dict() == {"HC05": -348.0, "HC06": 48.0}
    assert aug["total_pp"].sum() == 1000 + 40 + 7 + 9 + 200          # the formulary, return, S125, BH lines are out
    # Sep: HCG's Aug pair (master's lowest price), the master row, our rule, unpriced, none
    got = sep.set_index("material")[["suggestion_source", "status", "suggested_material", "sug_price",
                                     "price_source", "rule_level"]]
    assert got.loc["200010"].iloc[:5].tolist() == [O.HCG_SHEET_PAIR, O.COUNTED, "210010", 45.0,
                                                   O.PRICE_LIST_LOWEST]
    assert pd.isna(got.loc["200010", "rule_level"])
    assert got.loc["200060"].iloc[:5].tolist() == [O.HCG_LIST, O.COUNTED, "210010", 48.0, O.PRICE_LIST_ROW]
    assert got.loc["200080"].tolist() == [O.FORMULARY_RULE, O.COUNTED, "300002", 12.0, O.PRICE_CASH_TPA, "A"]
    assert got.loc["200020"].iloc[:3].tolist() == [O.HCG_SHEET_PAIR, O.UNPRICED, "210020"]
    assert got.loc["200090"].iloc[:2].tolist() == [O.NONE, O.NO_ALTERNATIVE]
    # HCG's own tables, and the report
    assert read_table(tmp_path / "data", C.HCG_SUMMARY)["loss"].tolist() == [-348.0, 48.0]
    assert len(read_table(tmp_path / "data", C.HCG_PAIRS)) == 3
    assert read_table(tmp_path / "data", C.HCG_PARKED)["lines"].tolist() == [1]
    v = json.loads((tmp_path / "data" / C.VALIDATION_FILE).read_text())
    assert v["periods"] == rep["periods"] == [AUG, SEP] and v["tables"][C.OOF]["rows"] == len(t)
    a = v["months"][AUG]
    assert [f["lines"] for f in a["funnel"]] == [9, 8, 8, 7, 6, 6]
    assert a["oof"]["hcg"]["max_centre_diff"] == 0.0 and a["oof"]["kr"]["diff_vs_email"] == \
        {"actual_margin": 0.0, "suggested_margin": 0.0, "margin_loss": 0.0}
    assert a["oof"]["no_pair"]["lines"] == 1
    sim = a["oof"]["simulation"]                     # no earlier sheet: master and rule only
    assert sim["pair_months"] == [] and set(sim["by_source"]) == {O.NONE}
    assert v["months"][SEP]["oof"]["pair_months"] == [AUG] and v["months"][SEP]["oof"]["hcg"] is None
    assert v["oof"]["months"] == [AUG, SEP] and "guard" in v["oof"]["rules"]


def test_the_golden_guard_stops_the_publish(tmp_path):
    data, cache = tmp_path / "data", tmp_path / "cache"
    off = [AUG_SUMMARY[0], ("HC06", "DR", 200.0, 200.0, 120.0, 175.0)]          # HC06 loss 55, not 48
    with pytest.raises(BuildError, match=r"centre HC06 loss 48.00 vs HCG's Summary 55.00"):
        B.run(make_inputs(summary=off), data=data, cache=cache)
    assert not data.exists() and (cache / "purchase_validation.failed.json").exists()
    with pytest.raises(BuildError, match=r"KR margin_loss -348.00 vs HCG's email -346.00"):
        B.run(make_inputs(email=(1040.0, 692.0, -346.0)), data=data, cache=cache)
    assert not data.exists()
    B.run(make_inputs(email=None), data=data, cache=cache, dry_run=True)        # no email row: no check
    assert not data.exists() and (cache / "purchase_validation.dry-run.json").exists()


def test_a_month_hcg_emailed_but_whose_workbook_is_missing_is_not_published(tmp_path):
    # HCG's email has KR figures for Aug, so HCG made an Aug worksheet: without it, sources 2-5
    # would quietly stand in for HCG's own figures
    data, cache = tmp_path / "data", tmp_path / "cache"
    with pytest.raises(BuildError, match=r"2026-08: HCG's email has KR figures for this month but HCG's "
                                         r"out-of-formulary workbook was not read.*--without-hcg-sheet 2026-08"):
        B.run(make_inputs(sheet=False), data=data, cache=cache)
    assert not data.exists()
    # told to, the build publishes the month from sources 2-5 and says so
    rep = B.run(make_inputs(sheet=False, without_sheet={AUG}), data=data, cache=cache)
    aug = rep["months"][AUG]["oof"]
    assert aug["hcg"] is None and any("worksheet is missing" in w for w in aug["warnings"])
    t = read_table(data)
    assert O.HCG_SHEET not in set(t["suggestion_source"]) and (t["period"] == AUG).any()
    assert not (data / f"{C.HCG_SUMMARY}.parquet").exists() or read_table(data, C.HCG_SUMMARY).empty


def test_a_month_built_without_its_workbook_keeps_hcgs_published_figures(tmp_path):
    data = tmp_path / "data"
    B.run(make_inputs(), data=data, cache=tmp_path)
    before = {name: read_table(data, name) for name in C.HCG_TABLES}
    assert all(len(df) for df in before.values())
    B.run(make_inputs(periods=(AUG,), sheet=False, without_sheet={AUG}), data=data, cache=tmp_path)
    t = read_table(data)
    assert O.HCG_SHEET not in set(t.loc[t["period"] == AUG, "suggestion_source"])     # Aug from sources 2-5
    for name, df in before.items():                  # HCG's Aug figures are still HCG's
        pd.testing.assert_frame_equal(read_table(data, name), df)


def test_rebuilding_one_month_keeps_the_others(tmp_path):
    data = tmp_path / "data"
    B.run(make_inputs(), data=data, cache=tmp_path)
    before = read_table(data)
    sections = json.loads((data / C.VALIDATION_FILE).read_text())
    B.run(make_inputs(periods=(SEP,)), data=data, cache=tmp_path)
    after = read_table(data)
    pd.testing.assert_frame_equal(after, before)
    v = json.loads((data / C.VALIDATION_FILE).read_text())
    assert v["periods"] == [AUG, SEP] and "oof" in v["months"][AUG]
    # the build-wide sections describe every month published, not only the one rebuilt
    assert v["oof"]["months"] == [AUG, SEP]
    for section in ("oof", "prices", "generic"):
        assert v[section] == sections[section], section


def test_no_identifier_or_date_is_published(tmp_path):
    for col in ("gr_no", "PO_NO", "invoice_no"):
        with pytest.raises(BuildError, match="identifier columns"):
            B.check_columns("t", pd.DataFrame({col: ["1"]}))
    with pytest.raises(BuildError, match="date columns"):
        B.check_columns("t", pd.DataFrame({"received": pd.to_datetime(["2026-08-01"])}))
    B.run(make_inputs(), data=tmp_path, cache=tmp_path)
    for name in (C.OOF, C.HCG_SUMMARY, C.HCG_PAIRS, C.HCG_PARKED):
        cols = {c.upper() for c in read_table(tmp_path, name).columns}
        assert not cols & (B.IDENTIFIER_COLUMNS | {"GR_QTY", "GR_DATE"}), name


def test_every_published_value_is_in_the_vocabulary_the_api_labels():
    ok = pd.DataFrame({"status": [O.COUNTED, O.PARKED], "suggestion_source": [O.HCG_SHEET, O.FORMULARY_RULE],
                       "price_source": [O.PRICE_HCG_SHEET, None], "rule_level": [None, "B" + O.FUZZY_MARK]})
    assert O.vocabulary_problems(ok) == []
    bad = ok.assign(price_source=["hcg_sheet", "guessed"], rule_level=[None, "Z"])
    assert O.vocabulary_problems(bad) == [
        f"oof_purchase.price_source holds guessed, outside its vocabulary ({', '.join(O.PRICE_SOURCES)})",
        f"oof_purchase.rule_level holds Z, outside its vocabulary ({', '.join(O.VOCABULARY['rule_level'])})"]


def test_a_builder_cannot_publish_another_builders_table(tmp_path):
    twice = (("oof", B.build_oof), ("again", B.build_oof))
    with pytest.raises(ValueError, match="built twice"):
        B.run(make_inputs(), data=tmp_path, cache=tmp_path, builders=twice)


def test_months_outside_the_lists_year_abort(tmp_path):
    inputs = make_inputs()
    inputs.main.cash.fy = (2025, 2026)
    with pytest.raises(BuildError, match="CASH&TPA list is for FY 2025-26; it does not cover 2026-08"):
        B.run(inputs, data=tmp_path, cache=tmp_path)
