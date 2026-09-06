# tests/test_coverage.py — what a table does NOT contain, measured rather than assumed.
#
# An audit of every table found fact_consumption was not alone: fact_inventory carries 26 of
# 53 hospitals, and the 27 missing ones — the BACC cancer centres, the Triesta labs — have
# thousands of purchase-order lines each. So "which hospital holds the most stock" was
# answered over half the estate with nothing saying so.
from __future__ import annotations

from app.ai.coverage import disclosure, measure


def test_inventory_is_measured_as_half_the_estate():
    present, total, missing = measure("fact_inventory", "hospital")
    assert total == 53 and present == 26
    assert missing, "the missing hospitals should be named, not just counted"


def test_consumption_material_gap_is_measured():
    present, total, _ = measure("fact_consumption", "material")
    assert present < total * 0.5          # 11,225 of 24,931


def test_a_partial_table_is_disclosed_when_the_query_touches_that_entity():
    d = disclosure("SELECT plant, SUM(total_cost) FROM fact_inventory GROUP BY 1")
    assert d and "26 of 53 hospitals" in d
    assert "Triesta" in d or "BACC" in d   # named, so the reader can see who is absent


def test_missing_things_are_named_not_coded():
    d = disclosure("SELECT material, SUM(qty) FROM fact_consumption GROUP BY 1")
    assert d and not any(tok.strip("',").isdigit() and len(tok.strip("',")) == 6
                         for tok in d.split() if tok.strip("',").isdigit())


def test_a_well_covered_table_says_nothing():
    assert disclosure("SELECT plant, SUM(line_value) FROM mart_procurement GROUP BY 1") is None


def test_a_table_that_IS_a_filter_says_nothing():
    # kpi_near_expiry holds 3,997 materials because only 3,997 are near expiry, and it is a
    # clean subset of fact_inventory (zero orphans). Its name states the filter.
    assert disclosure("SELECT * FROM kpi_near_expiry") is None


def test_a_query_that_does_not_touch_the_entity_says_nothing():
    # a note about hospital coverage on a query that never mentions hospitals is noise
    assert disclosure("SELECT SUM(revenue) FROM sales_totals") is None


def test_vendors_are_never_reported_as_a_coverage_gap():
    # dim_vendor is a MASTER LIST: 3,576 registered, 2,251 ever bought from. That gap is the
    # correct state of the world, and calling it missing data is a lie in the other direction.
    d = disclosure("SELECT vendor_code, SUM(line_value) FROM mart_procurement GROUP BY 1")
    assert d is None or "vendor" not in d.lower()


def test_malformed_input_never_raises():
    assert disclosure("") is None
    assert measure("no_such_table", "material") == (0, 0, ())
    assert measure("fact_inventory", "not_a_kind") == (0, 0, ())


def test_the_note_forbids_using_coverage_as_a_filter():
    # the first wording taught the model to SCOPE answers to the covered subset: asked how
    # much we spend on items never sold, it restricted itself to materials that DO appear in
    # sales — the opposite of the question — and reported ₹6.53 Cr instead of ₹60.43 Cr
    d = disclosure("SELECT material, SUM(qty) FROM fact_consumption GROUP BY 1")
    assert d and "Do NOT turn it into a filter" in d
