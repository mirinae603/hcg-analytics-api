# tests/test_generic_bridge.py — the molecule-to-brand bridge the warehouse already had.
#
# "How much pembrolizumab did we consume?" resolved PEMBROLIZUMAB correctly as a GENERIC and
# was answered "there is no direct match for PEMBROLIZUMAB in the catalog" — because no fact
# table carries a generic_name column and nothing said dim_material bridges it.
#
# This is the synonym problem people reach for embeddings to solve. It did not need them:
# dim_material holds generic_name beside material_desc, and PEMBROLIZUMAB is KEYTRUDA.
from __future__ import annotations

from app.ai.resolve import brief, generic_materials, resolve


def test_a_molecule_resolves_to_its_brands():
    assert "KEYTRUDA 100MG INJ VIAL" in generic_materials("PEMBROLIZUMAB")


def test_a_molecule_with_several_brands_returns_them_all():
    brands = generic_materials("TRASTUZUMAB")
    assert len(brands) > 1


def test_the_generic_is_typed_as_a_generic_not_a_material():
    ents = resolve("how much pembrolizumab did we consume?")["entities"]
    assert any(e["text"] == "PEMBROLIZUMAB" and e["kind"] == "generic" for e in ents)


def test_the_brief_hands_over_the_brands():
    b = brief("how much pembrolizumab did we consume?")
    assert "KEYTRUDA 100MG INJ VIAL" in b
    assert "appears in NO fact table" in b
    assert "never report the molecule as" in b


def test_an_unknown_molecule_returns_nothing_rather_than_guessing():
    assert generic_materials("NOTAREALMOLECULE") == ()
