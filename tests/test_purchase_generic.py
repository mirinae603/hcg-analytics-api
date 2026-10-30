# tests/test_purchase_generic.py — generic vs branded (app/etl/purchase/generic.py) on tiny made-up data.
#
# Branded = a product of an originator company, read from the GRN manufacturer through
# app/data/reference/originator_manufacturers.csv, with material overrides in
# generic_overrides.csv; everything else is generic. Each side is split by the month's own
# Item Master class, and the originator brands are priced against the cheapest generic of the
# same product that month. Company and product names are real (HCG's makers and brands); the
# lines are made up. The committed reference files are checked against the research that chose
# the definition, and the last section checks the committed table against the research's
# figures (skipped until `python -m app.etl.purchase` has published).
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.etl.formulary import classify as K
from app.etl.formulary.discover import BuildError
from app.etl.purchase import build as B
from app.etl.purchase import config as C
from app.etl.purchase import generic as Gn
from app.etl.purchase import prices as P
from tests.test_purchase_prices import AUG, JUL, frame, item_master, line, make_inputs

ORIGINATORS = """pattern,company,reason
MSD,MSD,originator
ASTRAZENECA,AstraZeneca,originator
ASTRA ZENECA,AstraZeneca,another spelling
J&J,Johnson & Johnson,originator
MERCK,Merck,originator
BMS,Bristol Myers Squibb,originator
ELI LILLY,Eli Lilly,originator
ROCHE,Roche,originator
BEICENE,BeiGene,SAP spelling
"""
OVERRIDES = """material,class,reason
100714,branded,VECTIBIX: Amgen's brand sold by Dr. Reddy's
200999,generic,an originator's unbranded generic
"""


def write_reference(folder: Path, originators=ORIGINATORS, overrides=OVERRIDES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if originators is not None:
        (folder / Gn.ORIGINATOR_FILE).write_text(originators)
    if overrides is not None:
        (folder / Gn.OVERRIDES_FILE).write_text(overrides)
    return folder


# --- the definition -------------------------------------------------------------------------
def test_name_words():
    assert Gn.name_words("DR.REDDY'S") == ("DR", "REDDYS")
    assert Gn.name_words("J & J") == Gn.name_words("J&J") == ("J", "J")
    assert Gn.name_words(" beicene  pvt ltd ") == ("BEICENE", "PVT", "LTD")
    assert Gn.name_words(None) == Gn.name_words(float("nan")) == Gn.name_words("Blank") == ()


@pytest.mark.parametrize("manufacturer, company", [
    ("MSD", "MSD"), ("ASTRAZENECA", "AstraZeneca"), ("ASTRA ZENECA", "AstraZeneca"),
    ("J & J", "Johnson & Johnson"), ("J&J", "Johnson & Johnson"), ("BEICENE PVT LTD", "BeiGene"),
    ("ELI LILLY", "Eli Lilly"), ("MERCK", "Merck"),
    ("LILLY", None),                    # the pattern is ELI LILLY: words, not letters, must match
    ("ROCHESTER LABS", None),           # ROCHE is a word, not a prefix of one
    ("MSDX PHARMA", None), ("DR.REDDY'S", None), (None, None), ("", None),
])
def test_originator_patterns_match_whole_words(tmp_path, manufacturer, company):
    d = Gn.load_definition(write_reference(tmp_path))
    o = d.originator(manufacturer)
    assert (o.company if o else None) == company


def test_classify_override_first(tmp_path):
    d = Gn.load_definition(write_reference(tmp_path))
    assert d.classify("100177", "MERCK") == Gn.Classification(Gn.BRANDED, Gn.ORIGINATOR_LIST, "Merck")
    assert d.classify("100714", "DR.REDDY'S") == Gn.Classification(Gn.BRANDED, Gn.OVERRIDE, None)
    assert d.classify("200999", "MSD") == Gn.Classification(Gn.GENERIC, Gn.OVERRIDE, "MSD")
    assert d.classify("102402", "LUPIN") == Gn.Classification(Gn.GENERIC, Gn.DEFAULT, None)
    assert d.classify("102402", None) == Gn.Classification(Gn.GENERIC, Gn.DEFAULT, None)
    assert d.overrides["100714"].reason.startswith("VECTIBIX")


@pytest.mark.parametrize("originators, overrides, error", [
    (None, OVERRIDES, "reference file not found"),
    (ORIGINATORS, None, "reference file not found"),
    ("pattern,company\nMSD,MSD\n", OVERRIDES, "needs the columns pattern, company, reason"),
    ("pattern,company,reason\n,MSD,x\n", OVERRIDES, "line 2: needs a pattern and a company"),
    ("pattern,company,reason\nMSD,,x\n", OVERRIDES, "line 2: needs a pattern and a company"),
    ("pattern,company,reason\nJ&J,J,x\nJ & J,J,y\n", OVERRIDES, "line 3: pattern J & J is already on line 2"),
    (ORIGINATORS, "material,class,reason\n100714,innovator,x\n", "needs a material and a class"),
    (ORIGINATORS, "material,class,reason\n,branded,x\n", "needs a material and a class"),
    (ORIGINATORS, "material,class,reason\n100714.0,branded,x\n100714,generic,y\n", "material 100714 listed twice"),
])
def test_bad_reference_files_stop_the_build(tmp_path, originators, overrides, error):
    with pytest.raises(BuildError, match=error):
        Gn.load_definition(write_reference(tmp_path, originators, overrides))


# The research (Aug-26 GRN) classed these 32 GRN manufacturers as originator companies; the
# committed list must find exactly them among HCG's makers, and none of the others below.
RESEARCH_ORIGINATORS = [
    "ABBIVE", "ALCON", "ALLERGAN", "ASTELLAS", "ASTRAZENECA", "AVENTIS", "BAYER", "BEICENE PVT LTD",
    "BMS", "BOEHRINGER", "BRISTOL MYERS SQUIBB", "EISAI", "ELI LILLY", "FERRING", "GENZYME", "GSK",
    "J&J", "JANSSEN", "LILLY", "MENARINI", "MENARINI INDIA PVT LTD", "MERCK", "MSD", "NOVARTIS",
    "NOVO NORDISK", "ORGANON", "PFIZER", "ROCHE", "SANOFI", "SERVIER INDIA PRIVATE LIMITED", "TAKEDA", "UCB"]
NOT_ORIGINATORS = [
    "ABBOTT", "FRESENIUS KABI", "BAXTER", "B BRAUN", "OTSUKA PHARMACEUTICAL PVT LTD", "SANDOZ",
    "ATTIS GLOBAL PHARMA", "JAN AUSHADHI", "DR.REDDY'S", "INTAS", "ZYDUS", "RELIANCE", "SUN", "LUPIN",
    "GEN", "COMMON", "HETERO", "BIOCON", "MSN", "AMNEAL HEALTHCARE PVT LTD", "GE HEALTHCARE"]


def test_the_committed_definition():
    d = Gn.load_definition(settings.REFERENCE_DIR)
    assert all(d.originator(n) for n in RESEARCH_ORIGINATORS)
    assert not [n for n in NOT_ORIGINATORS if d.originator(n)]
    assert {m: o.cls for m, o in d.overrides.items()} == {"100714": Gn.BRANDED, "102724": Gn.BRANDED,
                                                         "102881": Gn.BRANDED}
    assert all(o.reason for o in d.overrides.values()) and all(o.reason for o in d.originators)


# --- formulary class and cost impact ---------------------------------------------------------------
def test_formulary_class_of_the_month():
    ims = {AUG: item_master([("100177", "FORMULARY", "P-SCH", None, None),
                             ("100913", "FORMULARY", "R-OPD", None, None),
                             ("200010", "FORMULARY", "SCH", None, None),
                             ("200020", "FORMULARY", "OPD", None, None),
                             ("200030", "NON FORMULARY", None, None, None),
                             ("200040", "OUT OF FORMULARY", None, None, None)])}
    mats = ["100177", "100913", "200010", "200020", "200030", "200040", "200050"]
    df = frame([line(AUG, "HC05", m, 1, 1.0) for m in mats] + [line(JUL, "HC05", "100177", 1, 1.0)])
    got = Gn.with_formulary(df, ims)
    assert got["formulary_flag"].tolist()[:7] == ["FORMULARY"] * 4 + ["NON FORMULARY", "OUT OF FORMULARY",
                                                                       K.NOT_IN_MASTER]
    assert got["tier_group"].tolist()[:4] == [Gn.PREFERRED, Gn.RESTRICTED, Gn.SCHEME_ONLY, Gn.OTHER_TIER]
    assert got["tier_group"].iloc[4:].isna().all() and got["formulary_class"].iloc[1] == "FORMULARYR-OPD"
    assert got.iloc[7][["formulary_flag", "formulary_class", "tier", "tier_group"]].isna().all()  # no July master


NIVOLUMAB = [  # (plant, material, desc, maker, qty, total): unit price
    ("HC05", "101158", "OPDYTA 100MG INJ VIAL", "BMS", 10, 1_000_000.0),             # 100,000: branded
    ("HC06", "101158", "OPDYTA 100MG INJ VIAL", "BMS", 2, 160_000.0),                # 80,000: branded
    ("HC05", "102990", "TISHTHA 100MG INJ VIAL", "ZYDUS", 4, 99_252.0),              # 24,813: generic
    ("HC06", "102990", "TISHTHA 100MG INJ VIAL", "ZYDUS", 1, 30_000.0),              # 30,000
    ("HC13", "102991", "NIVOBIO 100MG INJ VIAL", "BIOCON", 1, 28_000.0),             # 28,000: dearer generic
    ("HC13", "101158", "OPDYTA 100MG INJ VIAL", "BMS", 1, 20_000.0),                 # below the generic: 0
]


def nivolumab_lines():
    rows = [line(AUG, pl, m, q, t, 110_000.0, desc=d, minor="NIVOLUMAB", maker=mk)
            for pl, m, d, mk, q, t in NIVOLUMAB]
    opdyta = dict(desc="OPDYTA 100MG INJ VIAL", minor="NIVOLUMAB", maker="BMS")
    rows += [
        line(AUG, "HC05", "101158", 3, 0.03, 110_000.0, **opdyta),          # a token price
        line(JUL, "HC05", "101158", 1, 100_000.0, 110_000.0, **opdyta),     # no generic bought in July
        line(AUG, "HC05", "100177", 1, 18_938.1, 20_500.0, desc="ERBITUX 100MG INJ VIAL",
             minor="CETUXIMAB", maker="MERCK"),                             # no generic of the product
        line(AUG, "HC05", "100714", 1, 95_000.0, 120_000.0, desc="VECTIBIX 100MG INJ 5ML VIAL",
             minor="PANITUMUMAB", maker="DR.REDDY'S"),                      # an override
    ]
    return rows


def classified(tmp_path, rows=None, ims=None):
    d = Gn.load_definition(write_reference(tmp_path))
    return Gn.classified_lines(P.priced_lines(frame(rows or nivolumab_lines()), ims or {}, {}), d, ims or {})


GENERIC_PRICE = 24_813.0      # TISHTHA's 4 vials at 24,813 hold over 10% of the 6 generic vials bought


def test_the_cheapest_generic_of_the_month(tmp_path):
    c = classified(tmp_path)
    best = Gn.cheapest_generics(c)
    assert best.to_dict("records") == [{"period": AUG, "molecule_key": "NIVOLUMAB|100MG|INJ|",
                                        "generic_material": "102990", "generic_desc": "TISHTHA 100MG INJ VIAL",
                                        "generic_price": GENERIC_PRICE}]


def test_one_odd_generic_purchase_is_not_the_benchmark(tmp_path):
    # as ceftazidime-avibactam in Aug-26: 15 vials of one generic at Rs 450 against 566 of
    # another at Rs 1,707.09 — the month's usual cheapest generic is the second
    niv = dict(minor="NIVOLUMAB")
    rows = [line(AUG, "HC05", "102990", 50, 1_250_000.0, 110_000.0, desc="TISHTHA 100MG INJ VIAL", maker="ZYDUS", **niv),
            line(AUG, "HC31", "102991", 2, 4_000.0, 110_000.0, desc="NIVOBIO 100MG INJ VIAL", maker="BIOCON", **niv),
            line(AUG, "HC05", "101158", 10, 1_000_000.0, 110_000.0, desc="OPDYTA 100MG INJ VIAL", maker="BMS", **niv)]
    c = classified(tmp_path, rows)
    best = Gn.cheapest_generics(c).iloc[0]
    assert (best["generic_material"], best["generic_price"]) == ("102990", 25_000.0)      # not NIVOBIO's 2,000
    assert c.loc[c["material"] == "101158", "extra_vs_generic"].sum() == pytest.approx((100_000 - 25_000) * 10)


def test_extra_paid_for_originator_brands(tmp_path):
    c = classified(tmp_path).set_index(["period", "plant", "material", "total_wo_tax"])
    extra = c["extra_vs_generic"]
    assert extra[(AUG, "HC05", "101158", 1_000_000.0)] == pytest.approx((100_000 - GENERIC_PRICE) * 10)
    assert extra[(AUG, "HC06", "101158", 160_000.0)] == pytest.approx((80_000 - GENERIC_PRICE) * 2)
    assert extra[(AUG, "HC13", "101158", 20_000.0)] == 0.0             # cheaper than the generic
    assert extra[(AUG, "HC05", "101158", 0.03)] == 0.0                 # a token price is not compared
    assert extra[(JUL, "HC05", "101158", 100_000.0)] == 0.0            # no generic bought in July
    assert extra[(AUG, "HC05", "100177", 18_938.1)] == 0.0             # no generic of cetuximab
    assert (extra.xs("102990", level="material") == 0).all()           # generics are not charged
    compared = c["compared_value"]
    assert compared.sum() == pytest.approx(1_000_000 + 160_000 + 20_000)
    assert c.loc[(AUG, "HC05", "100714", 95_000.0), "class_source"] == Gn.OVERRIDE


def test_aggregate_and_month_report(tmp_path):
    ims = {AUG: item_master([("101158", "FORMULARY", "P-SCH", "OPDYTA 100MG INJ VIAL", "NIVOLUMAB"),
                             ("102990", "OUT OF FORMULARY", None, "TISHTHA 100MG INJ VIAL", "NIVOLUMAB"),
                             ("102991", "NON FORMULARY", None, "NIVOBIO 100MG INJ VIAL", "NIVOLUMAB"),
                             ("100177", "FORMULARY", "R", "ERBITUX 100MG INJ VIAL", "CETUXIMAB"),
                             ("100714", "FORMULARY", "R", "VECTIBIX 100MG INJ 5ML VIAL", "PANITUMUMAB")])}
    t = Gn.aggregate(classified(tmp_path, ims=ims))
    assert list(t.columns) == Gn.GENERIC_COLUMNS and t[Gn.GROUP_KEYS].duplicated().sum() == 0
    assert t["lines"].sum() == 10 and t["lines"].dtype == "int64" and t["class"].dtype == "str"
    opdyta = t[(t["period"] == AUG) & (t["material"] == "101158") & (t["plant"] == "HC05")].iloc[0]
    assert (opdyta["class"], opdyta["class_source"], opdyta["originator_company"]) == \
        (Gn.BRANDED, Gn.ORIGINATOR_LIST, "Bristol Myers Squibb")
    assert (opdyta["lines"], opdyta["value"], opdyta["generic_material"]) == (2, 1_000_000.03, "102990")
    assert (opdyta["formulary_flag"], opdyta["tier"], opdyta["tier_group"]) == ("FORMULARY", "P-SCH", Gn.PREFERRED)
    assert t.loc[t["period"] == JUL, "formulary_flag"].isna().all()
    rep = Gn.month_report(t[t["period"] == AUG])
    branded = 1_000_000.03 + 160_000 + 20_000 + 18_938.1 + 95_000
    assert rep["by_class"][Gn.BRANDED]["value"] == pytest.approx(branded)
    assert rep["by_class"][Gn.GENERIC]["value"] == pytest.approx(99_252 + 30_000 + 28_000)
    assert rep["by_source"][Gn.OVERRIDE] == {"lines": 1, "value": 95_000.0}
    assert rep["overrides"] == [{"material": "100714", "material_desc": "VECTIBIX 100MG INJ 5ML VIAL",
                                 "class": Gn.BRANDED, "value": 95_000.0}]
    assert rep["formulary"] == {Gn.BRANDED: {"FORMULARY": pytest.approx(branded)},
                                Gn.GENERIC: {"NON FORMULARY": 28_000.0, "OUT OF FORMULARY": 129_252.0}}
    assert rep["tiers"] == {Gn.BRANDED: {Gn.PREFERRED: pytest.approx(1_180_000.03),
                                         Gn.RESTRICTED: pytest.approx(113_938.1)}}
    ci = rep["cost_impact"]
    assert ci["extra_vs_generic"] == pytest.approx((100_000 - GENERIC_PRICE) * 10 + (80_000 - GENERIC_PRICE) * 2,
                                                   abs=0.01)
    assert ci["compared_value"] == 1_180_000.0
    assert [r["material"] for r in ci["top"]] == ["101158"]
    assert ci["top"][0]["cheapest_generic"] == "TISHTHA 100MG INJ VIAL"
    assert Gn.month_report(t[t["period"] == JUL])["formulary"] is None            # no Item Master: no split


# --- the build ----------------------------------------------------------------------------------
def test_the_build_publishes_purchase_generic(tmp_path):
    ref = write_reference(tmp_path / "ref")
    inputs = make_inputs(nivolumab_lines(), reference=str(ref))
    rep = B.run(inputs, data=tmp_path / "data", cache=tmp_path / "cache",
                builders=(("prices", B.build_prices), ("generic", B.build_generic)))
    t = pd.read_parquet(tmp_path / "data" / f"{C.GENERIC}.parquet")
    assert list(t.columns) == Gn.GENERIC_COLUMNS and t["lines"].sum() == 10
    assert (tmp_path / "data" / f"{C.PRICES}.parquet").exists()
    B.check_columns(C.GENERIC, t)
    v = json.loads((tmp_path / "data" / C.VALIDATION_FILE).read_text())
    assert v["months"][AUG]["generic"]["cost_impact"]["extra_vs_generic"] == \
        rep["months"][AUG]["generic"]["cost_impact"]["extra_vs_generic"] > 0
    g = v["generic"]
    assert g["estimated"] is True and g["originator_patterns"] == 9 and g["originator_companies"] == 8
    assert g["manufacturers_counted_as_originators"] == {"BMS": "Bristol Myers Squibb", "MERCK": "Merck"}
    assert [o["material"] for o in g["overrides"]] == ["100714", "200999"]


def test_the_build_stops_without_the_definition(tmp_path):
    inputs = make_inputs(nivolumab_lines(), reference=str(write_reference(tmp_path / "ref", originators=None)))
    with pytest.raises(BuildError, match="originator_manufacturers.csv"):
        B.run(inputs, data=tmp_path / "data", cache=tmp_path / "cache",
              builders=(("generic", B.build_generic),))
    assert not (tmp_path / "data").exists()


# --- the committed table ------------------------------------------------------------------------
# The research that chose the definition (Aug-26 GRN, the purchase scope) found, with the
# originator list alone: branded Rs 22.73 Cr of Rs 45.58 Cr, split by the Item Master as below,
# and Rs 2.78 Cr paid for originator brands above the cheapest generic of the same molecule +
# strength + form (Jun 2.89, Jul 2.80), the cheapest generic being the lowest generic material's
# median price. The overrides then move the three licensed originator brands to branded. Our
# keys reproduced that within 1.5% (Aug Rs 2.79 Cr: molecule.py's salt words find generics the
# research's plain minor-group key missed — SUPACEF 1.5GM vs BACTILEM, FASLODEX vs AT-FULVESTRA,
# TASIGNA vs NINLIB). The benchmark is now the USUAL cheapest generic price (prices.usual_cheapest:
# the lowest at which at least 10% of the generic quantity was bought), so one small odd generic
# purchase no longer prices a month of branded ones: ZAVICEFTA is set against XAVITAZ at
# Rs 1,707.09 instead of 15 vials of VECAZ at Rs 450, ERBITUX against LUPITUX at Rs 8,100 instead
# of one vial of CETUXA at Rs 5,500. Aug falls to Rs 2.47 Cr; Jun and Jul move under 1%.
DATA = Path(settings.PURCHASE_DATA_DIR)
committed = pytest.mark.skipif(not (DATA / f"{C.GENERIC}.parquet").exists(),
                               reason="purchase tables not built — run python -m app.etl.purchase")
CRORE = 1e7
RESEARCH_ORIGINATOR_CR = {"2026-06": 30.645, "2026-07": 20.112, "2026-08": 22.731}
RESEARCH_AUG_BAR_CR = {("branded", "FORMULARY"): 22.527, ("branded", "NON FORMULARY"): 0.105,
                       ("branded", "OUT OF FORMULARY"): 0.099, ("generic", "FORMULARY"): 19.469,
                       ("generic", "NON FORMULARY"): 2.836, ("generic", "OUT OF FORMULARY"): 0.54}
RESEARCH_AUG_TIERS_CR = {("branded", Gn.PREFERRED): 11.018, ("branded", Gn.RESTRICTED): 11.509,
                         ("generic", Gn.PREFERRED): 14.577, ("generic", Gn.RESTRICTED): 2.956,
                         ("generic", Gn.SCHEME_ONLY): 1.79, ("generic", Gn.OTHER_TIER): 0.146}
EXTRA = {"2026-06": 28_824_471.96, "2026-07": 27_804_660.82, "2026-08": 24_702_552.55}
AUG_TOTAL, AUG_BRANDED = 455_760_310.08, 235_638_720.20


@pytest.fixture(scope="module")
def generic_table():
    return pd.read_parquet(DATA / f"{C.GENERIC}.parquet")


@pytest.fixture()
def research_months(generic_table):
    missing = sorted(set(RESEARCH_ORIGINATOR_CR) - set(generic_table["period"]))
    if missing:
        pytest.skip(f"not built: {', '.join(missing)}")
    return sorted(RESEARCH_ORIGINATOR_CR)


@committed
def test_the_research_split_is_reproduced(generic_table, research_months):
    for p in research_months:
        t = generic_table[generic_table["period"] == p]
        by_list = t["class_source"].eq(Gn.ORIGINATOR_LIST)
        assert round(t.loc[by_list, "value"].sum() / CRORE, 3) == RESEARCH_ORIGINATOR_CR[p], p
    aug = generic_table[generic_table["period"] == "2026-08"]
    side = aug["class_source"].eq(Gn.ORIGINATOR_LIST).map({True: "branded", False: "generic"})
    bar = (aug.groupby([side, "formulary_flag"])["value"].sum() / CRORE).round(3).to_dict()
    tiers = (aug.groupby([side, "tier_group"])["value"].sum() / CRORE).round(3).to_dict()
    assert bar == RESEARCH_AUG_BAR_CR and tiers == RESEARCH_AUG_TIERS_CR


@committed
def test_the_overrides_move_the_licensed_originator_brands(generic_table, research_months):
    aug = generic_table[generic_table["period"] == "2026-08"]
    assert aug["value"].sum() == pytest.approx(AUG_TOTAL, abs=0.01)
    branded = aug["class"].eq(Gn.BRANDED)
    assert aug.loc[branded, "value"].sum() == pytest.approx(AUG_BRANDED, abs=0.01)
    moved = aug[aug["class_source"].eq(Gn.OVERRIDE)]
    assert set(moved["material"]) == {"100714", "102724", "102881"}
    assert aug.loc[branded, "value"].sum() - aug.loc[aug["class_source"].eq(Gn.ORIGINATOR_LIST), "value"].sum() == \
        pytest.approx(moved["value"].sum())


@committed
def test_cost_impact(generic_table, research_months):
    for p in research_months:
        t = generic_table[generic_table["period"] == p]
        assert t["extra_vs_generic"].sum() == pytest.approx(EXTRA[p], abs=1.0), p
        assert (t.loc[t["class"].eq(Gn.GENERIC), ["extra_vs_generic", "compared_value"]] == 0).all().all(), p
    aug = generic_table[generic_table["period"] == "2026-08"]
    top = aug.groupby("material_desc")["extra_vs_generic"].sum().sort_values(ascending=False)
    assert [round(v / CRORE, 3) for v in top.iloc[:2]] == [0.692, 0.542]        # OPDYTA 100MG, ERBITUX
    assert list(top.index[:2]) == ["OPDYTA 100MG INJ VIAL", "ERBITUX 100MG INJ VIAL"]
    zav = aug[aug["material_desc"].eq("ZAVICEFTA 2GM/0.5MG INJ VIAL")]
    assert set(zav["generic_desc"]) == {"XAVITAZ 2GM/0.5MG INJ VIAL"} and set(zav["generic_price"]) == {1_707.09}


@committed
def test_the_published_generic_table_is_an_aggregate(generic_table):
    B.check_columns(C.GENERIC, generic_table)
    assert list(generic_table.columns) == Gn.GENERIC_COLUMNS
    assert generic_table[["period", "plant", "material"]].duplicated().sum() == 0
    assert set(generic_table["class"]) <= set(Gn.CLASSES)
    assert set(generic_table["class_source"]) <= set(Gn.CLASS_SOURCES)
    report = json.loads((DATA / C.VALIDATION_FILE).read_text())
    no_master = [p for p in report["months"] if p not in report["oof"]["months"]]
    assert generic_table.loc[generic_table["period"].isin(no_master), "formulary_flag"].isna().all()
