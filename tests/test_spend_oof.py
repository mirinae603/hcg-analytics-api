# tests/test_spend_oof.py — the shared description parsers (app/etl/spend/oof.py) and the CASH&TPA
# price list (app/etl/spend/masters.load_price_list), on tiny made-up data.
#
# The out-of-formulary margin itself is no longer a spend (sales) table: it follows HCG's purchase
# (GRN) method in app/etl/purchase, which builds on these keys and prices (tests/test_purchase_*.py).
# Every expected key and price below is worked out by hand.
from __future__ import annotations

import pytest

from app.etl.spend import oof as O
from app.etl.spend.masters import PricedBrand, load_price_list
from tests.test_spend_pipeline import write_scheme_list


# --- 1. the description keys ---------------------------------------------------------------

@pytest.mark.parametrize("desc, key", [
    ("ZAVIEDGE 2GM/0.5MG INJ VIAL", "0.5MG|2000MG"),
    ("XAVITAZ 2 GM / 0.5 MG INJ VIAL", "0.5MG|2000MG"),         # spaces and GM vs MG agree
    ("CEFTRIAXONE 1000MG INJ", "1000MG"),
    ("CEFTRIAXONE 1 GM INJ", "1000MG"),                          # 1 GM = 1000 MG
    ("BUDECORT 200MCG ROTACAP", "0.2MG"),                        # 1000 MCG = 1 MG
    ("ADRIB 10MG/5ML INJ", "10MG"),                              # a dose fixes the pack
    ("PARACETAMOL 1 GM/100 ML INFUSION", "1000MG"),              # a dose in a stated volume, too
    ("NS 0.9% (DENIS) 250ML PB", "0.9%|250ML"),                  # a % alone does not
    ("DEXTROSE 25% 1 LTR", "1000ML|25%"),                        # 1 LTR = 1000 ML
    ("INSULIN 100IU/ML 3ML PEN", "100IU|3ML"),                   # nor a dose per mL
    ("KCL 2 MEQ/ML 10ML AMP", "10ML|2MEQ"),
    ("TUSQ-DX SYR 100ML BOTTLE", "100ML"),                       # no dose: the volume
    ("EPOFIT 4000 UNITS PFS", "4000IU"),                         # UNITS is a unit
    ("WEPOX 4000 IU PFS", "4000IU"),
    ("HEPARIN 25000 UNITS 5ML VIAL", "25000IU"),
    ("DOLO 650 TAB", "650"),                                     # no unit: the bare number
    ("VIT B12 TAB", ""),                                         # part of a word, not a strength
    ("DOLO TAB 10'S", ""),                                       # a pack count, not a strength
    (None, ""),
])
def test_strength_key(desc, key):
    assert O.strength_key(desc) == key


@pytest.mark.parametrize("desc, form", [
    ("ZAVIEDGE 2GM INJ VIAL", "INJ"), ("AMPICILLIN 500MG CAP", "CAP"), ("CAPECITABINE 500MG TAB", "TAB"),
    ("ASTHALIN 2.5MG RESP 2.5ML AMP", "NEB"), ("ASTHALIN NEBULISER SOLN", "NEB"),
    ("BUDECORT 200MCG ROTACAP", "DPI"), ("BUDECORT 200MCG INHALER", "MDI"), ("SEROFLO 250 EVOHALER", "MDI"),
    ("TUSQ-DX SYR 100ML BOTTLE", "ORAL LIQUID"), ("NS 0.9% 250ML PB", "INJ"),
    ("ONDANSETRON 4MG/2ML AMP", "INJ"), ("FILGRASTIM 300MCG PFS", "INJ"),
    ("BETADINE OINTMENT", "TOPICAL"), ("CIPLOX EYE DROPS", "DROPS"), ("NASIVION SPRAY", "SPRAY"),
    ("ORS SACHET", "POWDER"), ("DULCOLAX SUPPOSITORY", "SUPP"), ("FENTANYL 25MCG PATCH", "PATCH"),
    ("VALENTAS 50MG", ""), (None, ""),
])
def test_form_key(desc, form):
    assert O.form_key(desc) == form


@pytest.mark.parametrize("desc, kind", [
    ("TRAMADOL 100MG SR TAB", "MR"), ("METFORMIN 500MG XR TAB", "MR"), ("NIFEDIPINE 20MG RETARD TAB", "MR"),
    ("TRAMADOL 100MG TAB", ""), ("DOXORUBICIN LIPOSOMAL 20MG INJ", "LIPO"),
    ("PEGYLATED LIPOSOMAL DOXORUBICIN 20MG", "LIPO"), ("PACLITAXEL PROTEIN BOUND 100MG INJ", "NAB"),
    ("ABRAXANE ALBUMIN BOUND 100MG", "NAB"), ("NAB-PACLITAXEL 100MG", "NAB"),
    ("LIPODOX 50MG INJ VIAL", "LIPO"),                      # liposomal, though the name does not say
    ("ADRIM 50MG INJ VIAL", ""), ("LIPOSTERIL 20% INJ 100ML BOTTLE", ""), (None, ""),
])
def test_kind_key(desc, kind):
    assert O.kind_key(desc) == kind


def test_one_medicine_from_two_makers_gets_one_key():
    a, b = "ZAVIEDGE 2GM/0.5MG INJ VIAL", "XAVITAZ 2 GM / 0.5 MG INJECTION"
    assert (O.strength_key(a), O.form_key(a), O.kind_key(a)) == (O.strength_key(b), O.form_key(b), O.kind_key(b))
    plain, sr = "TRAMADOL 100MG TAB", "TRAMADOL 100MG SR TAB"           # same dose, not a substitute
    assert O.strength_key(plain) == O.strength_key(sr) and O.kind_key(plain) != O.kind_key(sr)


# --- 2. the CASH&TPA price list ------------------------------------------------------------

# CASH&TPA rows: used, price, MRP, GST, category, SHARING, suggested, its price, MRP, GST
PRICE_ROWS = [
    (101001, 1000.0, 1260.0, 5, "R", 0.5, 101005, 300.0, 1155.0, 5),     # a cash switch too
    (101005, 300.0, 1155.0, 5, "P", 0.5, None, None, None, None),        # no suggestion: still priced
    (101006, 500.0, 2100.0, 5, "P-0PD", 1.0, None, None, None, None),    # HCG's typo canonicalised
    (101010, 50.0, None, 5, "P", 1.0, None, None, None, None),           # no MRP: not priced
    (101005, 999.0, 1999.0, 5, "P", 0.5, None, None, None, None),        # listed twice: first row kept
]


def test_the_price_list_keeps_every_priced_brand(tmp_path):
    path = tmp_path / "Copy of Onco Formulary -Spend Analysis Main Sheet_11.08.26.xlsx"
    write_scheme_list(path, cash=PRICE_ROWS)
    p = load_price_list(path)
    assert sorted(p) == ["101001", "101005", "101006"]
    assert p["101005"] == PricedBrand(material="101005", material_desc="CASH NAME 101005", price=300.0,
                                      mrp=1155.0, gst=5.0, category="P")
    assert p["101005"].unit_margin == pytest.approx(800.0)                 # 1155 / 1.05 - 300
    assert (p["101001"].category, p["101006"].category) == ("R", "P-OPD")
    assert p["101006"].unit_margin == pytest.approx(1500.0)                # 2100 / 1.05 - 500
