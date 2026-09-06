# tests/test_resolve.py — understanding the question BEFORE querying it.
#
# Every wrong answer this assistant produced came from acting on a word before typing it:
# "MSD" answered as a stationery sticker called STICKER-MSDS while MSD the manufacturer has
# 614 procurement lines; "Bangalore" answered as a sales filter that cannot exist, because a
# city lives only in dim_plant.plant_name and that column shares no rows with any sales
# table. Naming the things first is the step a human never skips.
from __future__ import annotations

from app.ai import resolve


# ── entity typing: the bug that started this ─────────────────────────────────
def test_msd_is_typed_as_a_manufacturer_not_an_item():
    r = resolve.resolve("Show me the procurement history for MSD products")
    msd = [e for e in r["entities"] if e["text"].upper() == "MSD"]
    assert msd, "MSD must resolve"
    assert msd[0]["kind"] == "manufacturer"
    assert msd[0]["column"] == "manufacturer_desc"
    # and it must NOT come back as the stationery sticker it shares a substring with
    assert not any("STICKER" in e["text"].upper() for e in r["entities"])


def test_a_real_product_is_typed_as_a_material():
    r = resolve.resolve('Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"')
    mats = [e for e in r["entities"] if e["kind"] == "material"]
    assert any("KEYTRUDA" in e["text"].upper() for e in mats)


def test_a_city_is_typed_as_a_city_and_carries_its_constraint():
    r = resolve.resolve("What are the top selling drugs in our Bangalore hospitals?")
    assert "Bangalore" in r["cities"]
    # and the brief must say where cities live, since that is the whole trap
    b = resolve.brief("What are the top selling drugs in our Bangalore hospitals?")
    assert "dim_plant.plant_name" in b
    assert "HC05" in b        # the hospitals the city actually covers


# ── the false positives that made a first version useless ────────────────────
def test_measure_words_do_not_resolve_to_companies_that_contain_them():
    # "sales trend" matched a vendor literally named "Pharma Sales"
    r = resolve.resolve('Show me the sales trend for "KEYTRUDA 100MG INJ VIAL"')
    assert not any(e["kind"] == "vendor" for e in r["entities"]), r["entities"]


def test_a_dosage_form_does_not_resolve_to_a_manufacturer():
    # "how many tablets" matched a manufacturer named "TABLETS INDIA"
    r = resolve.resolve("How many tablets do we stock?")
    assert not any("TABLETS INDIA" in e["text"].upper() for e in r["entities"])


def test_one_shared_token_is_not_enough_for_an_inexact_match():
    # a single overlapping word out of a value's several is a coincidence, and coincidences
    # are exactly how the wrong entity got picked
    r = resolve.resolve("what is our total stock value")
    assert all(e["exact"] or e["confidence"] >= 0.75 for e in r["entities"])


# ── intent: what is being measured, and at what grain ────────────────────────
def test_the_measure_is_identified():
    assert "revenue" in resolve.resolve("what were our sales last month")["measures"]
    assert "purchasing" in resolve.resolve("how much did we spend on procurement")["measures"]
    assert "expiry" in resolve.resolve("how much stock is expiring")["measures"]
    assert "lead_time" in resolve.resolve("which vendor has the worst lead time")["measures"]


def test_the_requested_grain_is_identified():
    assert "month" in resolve.resolve("show me the monthly trend")["grains"]
    assert "hospital" in resolve.resolve("break it down by hospital")["grains"]
    assert "vendor" in resolve.resolve("which supplier is biggest")["grains"]


def test_a_question_naming_nothing_resolves_to_nothing():
    # the resolver must stay silent rather than inventing an entity to talk about
    assert resolve.brief("hello") == ""


def test_the_brief_names_the_column_that_holds_each_value():
    b = resolve.brief("procurement for MSD")
    assert "manufacturer_desc" in b
