# tests/test_deep_grain.py — answering at the level the question asked for.
#
# "Which products move the most units?" was answered "M070-STATIONARY" — a CATEGORY.
# "Which manufacturer accounts for the most sales revenue?" was answered from PURCHASING.
# In both cases a finding that could not provide the requested breakdown became the
# headline, and one level up reads as an answer while being a different question.
from __future__ import annotations

from app.ai.deep.engine import _serves_grain


def test_a_category_result_does_not_satisfy_a_product_question():
    assert not _serves_grain({"columns": ["category", "qty"]}, "material")
    assert not _serves_grain({"columns": ["material_group", "units"]}, "material")


def test_a_product_result_does():
    assert _serves_grain({"columns": ["material_desc", "qty"]}, "material")
    assert _serves_grain({"columns": ["material", "units", "cost"]}, "material")


def test_a_bare_name_column_does_not_satisfy_the_product_grain():
    # the units-per-SKU KPI has a column literally called `name` holding CATEGORY labels
    # ("M070-STATIONARY"), so accepting `name` let the wrong answer pass the grain check
    assert not _serves_grain({"columns": ["name", "units", "share"]}, "material")


def test_vendor_and_manufacturer_are_not_interchangeable():
    assert not _serves_grain({"columns": ["vendor_name", "spend"]}, "manufacturer")
    assert _serves_grain({"columns": ["manufacturer_desc", "revenue"]}, "manufacturer")
    assert not _serves_grain({"columns": ["manufacturer", "revenue"]}, "vendor")


def test_hospital_accepts_either_naming():
    for col in ("hospital", "plant", "plant_name"):
        assert _serves_grain({"columns": [col, "value"]}, "hospital"), col


def test_a_time_grain_needs_a_time_column():
    assert _serves_grain({"columns": ["month", "revenue"]}, "month")
    assert not _serves_grain({"columns": ["material", "revenue"]}, "month")


def test_an_unknown_grain_never_blocks_anything():
    # the guard must fail OPEN: a grain we do not model is not a reason to reject a finding
    assert _serves_grain({"columns": ["anything"]}, "formulary")
    assert _serves_grain({"columns": ["anything"]}, "")
