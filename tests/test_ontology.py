# tests/test_ontology.py — the domain, derived instead of typed.
#
# The assistant knows this warehouse because ~800 literals were hand-written across 51
# constant tables: _MEASURES, _GRAINS, _STOP, _NOISE_TOKENS, _MEASURE_COLUMNS, GRAIN_COLUMNS,
# _TABLE_EVENT, _PLACEHOLDER. Each is a fact the system holds only because somebody typed it,
# and each is one release behind the moment a column is renamed.
#
# The measured case: on the OMG Property & Casualty schema, GPT-4 answered 16% of questions
# against the raw SQL schema, 54% against a knowledge-graph representation, and 72% once the
# ontology also CHECKED and repaired generated queries (arXiv 2311.07509 / 2405.11706).
#
# The test that matters is the last one: this ontology found, unaided, the trap that took a
# full day to find by hand.
from __future__ import annotations

import pytest

from app.ai import ontology as o

ONT = o.load()
pytestmark = pytest.mark.skipif(not ONT.get("columns"), reason="ontology not built yet")


def test_it_was_built_from_the_warehouse():
    assert len(ONT["columns"]) > 200
    assert ONT["tables"]


def test_it_found_the_disjoint_hospital_code_systems_by_itself():
    """The load-bearing test.

    dim_plant.plant and sales_by_hospital.hospital are both the hospital and share ZERO
    values, so no sales figure can be scoped to a city. Nobody told the ontology this; it
    classified both columns as the `hospital` entity and then measured the overlap.
    """
    pairs = [(d["a"], d.get("other", d["column"]), d["b"]) for d in ONT["disjoint"]]
    assert any("sales_by_hospital" in (a, b) and "dim_plant" in (a, b)
               for a, _c, b in pairs), pairs[:8]


def test_it_found_more_than_one_category_taxonomy():
    # `category` is a 134-value material group in one table and a 5-value business segment
    # ("Onco Drugs", "Lab", "Consumables") in another, with zero overlap
    assert any(d["column"] == "category" for d in ONT["disjoint"])


def test_non_additive_measures_are_identified():
    # summing a rate or a pre-computed average is the error that printed "₹85" for 85%
    na = o.non_additive()
    assert len(na) > 10
    assert any("_pct" in x or "rate" in x for x in na)


def test_measures_are_classified_by_business_event():
    # replaces the hand-written _TABLE_EVENT: sold vs bought vs issued vs held
    for event in ("sales", "procurement", "consumption", "stock"):
        assert o.measures_for(event=event), event


def test_entity_columns_are_derived():
    # replaces the hand-written GRAIN_COLUMNS
    for entity in ("material", "hospital"):
        assert len(o.columns_for_entity(entity)) >= 2, entity


def test_synonyms_are_generated_not_listed():
    assert len(o.synonyms()) > 100


# ── the verifier is what makes a generated ontology trustworthy ──────────────────────────
def test_a_non_numeric_column_cannot_be_called_a_measure():
    prof = {"tables": {"t": [{"table": "t", "column": "name", "rows": 10, "distinct": 9,
                              "null_pct": 0, "type": "VARCHAR", "uniqueness": 0.9,
                              "samples": ["x"]}]}, "links": []}
    _clean, rejected = o.verify({"columns": {"t.name": {"role": "measure"}}, "tables": {}}, prof)
    assert any("not numeric" in r for r in rejected)


def test_a_hallucinated_column_is_dropped():
    prof = {"tables": {"t": []}, "links": []}
    clean, rejected = o.verify({"columns": {"t.ghost": {"role": "measure"}}, "tables": {}}, prof)
    assert not clean["columns"] and any("no such column" in r for r in rejected)


def test_a_column_named_like_a_rate_cannot_claim_to_be_additive():
    prof = {"tables": {"t": [{"table": "t", "column": "margin_pct", "rows": 10, "distinct": 9,
                              "null_pct": 0, "type": "DOUBLE", "min": 0, "max": 1,
                              "uniqueness": 0.9, "all_zero": False, "samples": ["1"]}]},
            "links": []}
    clean, _ = o.verify(
        {"columns": {"t.margin_pct": {"role": "measure", "additive": True}}, "tables": {}}, prof)
    assert clean["columns"]["t.margin_pct"]["additive"] is False
