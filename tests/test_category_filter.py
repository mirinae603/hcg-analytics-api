# tests/test_category_filter.py — the global material-category dimension.
#
# The category column is DERIVED at the data_access._normalize choke point from
# dim_material.material_type (the only place HCG's material taxonomy exists), so
# these tests pin three things that are easy to break silently:
#   1. the buckets map exactly as specified and their stock value sums back to the
#      Rs 60.47 Cr portfolio total with NOTHING unmapped or dropped;
#   2. filtering genuinely subsets (and a category cut of a plant cut still adds up);
#   3. the UNFILTERED path is byte-identical — every existing dashboard number must
#      be untouched when the frontend sends no Category, which is the whole
#      backwards-compatibility contract of this change.
from __future__ import annotations

import pytest

from app.core import data_access as da

# Verified directly off the parquet (see the module docstring in data_access.py).
TOTAL_STOCK_VALUE = 604679341.2          # Rs 60.4679 Cr, == kpi_stock_value total
EXPECTED_SHARES = {                       # % of total stock value, verified
    "Onco Drugs": 41.67,
    "Consumables": 36.44,
    "Other Drugs": 17.03,
    "Non-Medical": 3.49,
    "Lab": 1.37,
}


@pytest.fixture(scope="module")
def stock_value():
    return da.load("kpi_stock_value")


# ---------- the bucket mapping ----------

def test_material_type_codes_map_to_the_specified_buckets():
    # The onco / non-onco drug split is the single most decision-relevant cut for an
    # oncology chain — it must never be collapsed into one "Pharma" bucket.
    assert da.category_of_material_type("ZOC-Medical Onco Drugs") == "Onco Drugs"
    assert da.category_of_material_type("ZNOC-Medical Non Onco Drugs") == "Other Drugs"
    assert da.category_of_material_type("ZMC-Medical Consumables") == "Consumables"
    for lab in ("ZLR-Laboratory Reagents", "ZLCL-Lab Calibrators", "ZLCO-Lab Controls"):
        assert da.category_of_material_type(lab) == "Lab"
    for nm in ("ZNMC-Non Medical Consumables", "ZNMA-Non Medical Asset", "ZMA-Medical Asset"):
        assert da.category_of_material_type(nm) == "Non-Medical"


def test_onco_and_non_onco_drugs_are_never_merged():
    assert da.category_of_material_type("ZOC-Medical Onco Drugs") != \
        da.category_of_material_type("ZNOC-Medical Non Onco Drugs")


@pytest.mark.parametrize("bad", [None, "", "   ", "nan", "None", "ZZZ-Something New"])
def test_null_or_unknown_material_type_is_labelled_unclassified_not_dropped(bad):
    # Never silently dropped and never silently folded into a real bucket.
    assert da.category_of_material_type(bad) == da.CATEGORY_UNCLASSIFIED


def test_every_material_gets_exactly_one_category():
    dm = da.load("dim_material")
    assert "category" in dm.columns
    assert dm["category"].notna().all()
    assert set(dm["category"].unique()) <= set(da.CATEGORIES)
    # the ~6,851 materials with a null material_type land in Unclassified
    assert int((dm["category"] == da.CATEGORY_UNCLASSIFIED).sum()) == 6851


# ---------- the buckets sum back to the portfolio total ----------

def test_category_shares_of_stock_value_match_ground_truth(stock_value):
    total = float(stock_value["stock_value_cost"].sum())
    assert total == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    by = stock_value.groupby("category", observed=True)["stock_value_cost"].sum()
    for name, share in EXPECTED_SHARES.items():
        assert float(by[name]) / total * 100 == pytest.approx(share, abs=0.01), name


def test_categories_sum_back_to_the_full_portfolio_with_nothing_unmapped(stock_value):
    total = float(stock_value["stock_value_cost"].sum())
    by = stock_value.groupby("category", observed=True)["stock_value_cost"].sum()
    assert float(by.sum()) == pytest.approx(total, rel=1e-9)
    # The ~6,851 unmapped materials hold ZERO stock value, so the filter is complete
    # on VALUE even though it is not on raw dim_material row count.
    assert float(by.get(da.CATEGORY_UNCLASSIFIED, 0.0)) == 0.0


def test_fact_inventory_derives_the_same_totals_by_a_different_path():
    # fact_inventory carries material_type natively, so it never touches the
    # material -> category map. Both paths must agree exactly, or one of them lies.
    inv = da.load("fact_inventory")
    a = inv.groupby("category", observed=True)["total_cost"].sum()
    b = da.load("kpi_stock_value").groupby("category", observed=True)["stock_value_cost"].sum()
    for name in EXPECTED_SHARES:
        assert float(a[name]) == pytest.approx(float(b[name]), rel=1e-9), name


# ---------- filter_category behaviour ----------

def test_filter_category_subsets_and_preserves_the_slice_total(stock_value):
    onco = da.filter_category(stock_value, "Onco Drugs")
    assert 0 < len(onco) < len(stock_value)
    assert set(onco["category"].unique()) == {"Onco Drugs"}
    assert float(onco["stock_value_cost"].sum()) == pytest.approx(251975495.01, abs=1.0)


@pytest.mark.parametrize("blank", [None, "", "All", "all", "ALL CATEGORIES"])
def test_no_category_means_no_filtering(stock_value, blank):
    assert da.resolve_category(blank) is None
    out = da.filter_category(stock_value, blank)
    assert len(out) == len(stock_value)
    assert float(out["stock_value_cost"].sum()) == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)


def test_raw_sap_code_is_accepted_as_an_alias(stock_value):
    assert da.resolve_category("ZOC") == "Onco Drugs"
    assert len(da.filter_category(stock_value, "ZOC")) == \
        len(da.filter_category(stock_value, "Onco Drugs"))


def test_unrecognised_category_yields_nothing_rather_than_everything(stock_value):
    # An unknown value must NEVER quietly hand back the whole portfolio under a
    # filter label the user believes is applied.
    assert len(da.filter_category(stock_value, "Pharma")) == 0


def test_filter_is_a_noop_on_tables_with_no_material_grain():
    # kpi_aging_distribution is pre-aggregated to plant x material_group x bucket and
    # has no material column, so no category column is derived for it at all — the
    # committed parquet's shape must be completely untouched.
    ad = da.load("kpi_aging_distribution")
    assert "category" not in ad.columns
    assert len(ad) == 6497
    assert len(da.filter_category(ad, "Onco Drugs")) == len(ad)


def test_existing_category_column_is_never_overwritten():
    # kpi_purchase_value already has its own (PO) `category` column — the derived
    # dimension must not clobber it.
    pv = da.load("kpi_purchase_value")
    assert pv["category"].nunique() > 20


def test_plant_and_category_filters_compose(stock_value):
    plant = da.filter_plant(stock_value, "AH01")
    both = da.filter_category(plant, "Onco Drugs")
    assert len(both) <= len(plant) <= len(stock_value)
    assert set(both["plant"].astype(str).unique()) == {"AH01"}
    # the per-plant category slices must add back up to that plant's total
    parts = sum(float(da.filter_category(plant, c)["stock_value_cost"].sum())
                for c in da.CATEGORIES)
    assert parts == pytest.approx(float(plant["stock_value_cost"].sum()), rel=1e-9)


# ---------- the API surface ----------

def test_meta_categories_lists_the_buckets_the_frontend_should_render(client):
    r = client.get("/meta/categories")
    assert r.status_code == 200, r.text
    body = r.json()
    keys = [c["key"] for c in body["categories"]]
    assert keys[0] == "All"                       # the default, always first
    # Unclassified IS offered. It holds zero stock, which is why an earlier version
    # hid it — but it holds ~50% of consumption cost and 10,614 of the 19,014 reorder
    # lines, so hiding it meant picking any category silently dropped over half the
    # reorder queue. A bucket is only hidden when it is empty in EVERY domain.
    assert keys[1:] == ["Onco Drugs", "Other Drugs", "Consumables", "Lab",
                        "Non-Medical", "Unclassified"]
    assert body["total_value"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    assert sum(c["share_pct"] for c in body["categories"][1:]) == pytest.approx(100.0, abs=0.1)


def test_offered_buckets_sum_back_to_all_on_reorder_and_consumption(client):
    """The property that was actually broken: the buckets the UI OFFERS must reconcile
    to the unfiltered total on every domain the filter applies to — not just on stock.

    The prior test only iterated da.CATEGORIES (which includes Unclassified) against a
    stock endpoint, so it passed while the real selector was dropping 10,614 reorder
    lines and Rs 33.6 Cr of consumption cost into an unreachable bucket.
    """
    offered = [c["key"] for c in client.get("/meta/categories").json()["categories"]
               if c["key"] != "All"]

    all_lines = client.get("/forecast/reorder-priority").json()["totals"]["reorder_lines"]
    summed = sum(client.get(f"/forecast/reorder-priority?Category={c}")
                 .json()["totals"]["reorder_lines"] for c in offered)
    assert summed == all_lines == 19014, (
        f"{all_lines - summed} reorder lines are unreachable from the selector")

    all_cost = client.get("/kpi/unit-sold-per-sku/insights").json()["totals"]["cost"]
    summed_cost = sum(client.get(f"/kpi/unit-sold-per-sku/insights?Category={c}")
                      .json()["totals"]["cost"] for c in offered)
    assert summed_cost == pytest.approx(all_cost, rel=1e-9)


def test_foreign_category_column_is_never_filtered_as_a_material_category(client):
    """kpi_purchase_value ships its OWN `category` (the PO taxonomy). Filtering a
    material category against it matched nothing and returned Rs 0, while
    monthly-purchase-value returned Rs 236.7 Cr for the same filter — two procurement
    money metrics flatly contradicting each other under one filter."""
    unfiltered = client.get("/kpi/purchase-value").json()
    filtered = client.get("/kpi/purchase-value?Category=Onco Drugs").json()
    rows_u = unfiltered if isinstance(unfiltered, list) else unfiltered.get("data", [])
    rows_f = filtered if isinstance(filtered, list) else filtered.get("data", [])
    assert len(rows_f) == len(rows_u) > 0, "purchase-value must not be zeroed by a category"


def test_dashboard_all_survives_a_category_with_no_rows(client):
    """`median() or 0` does not guard an empty frame — nan is truthy, so nan reached
    the JSON encoder and the endpoint 500'd. Unreachable until Category existed."""
    for cat in ("Unclassified", "Bogus Category"):
        r = client.get(f"/api/dashboard/all?Category={cat}")
        assert r.status_code == 200, f"{cat} -> {r.status_code}: {r.text[:200]}"


def test_meta_categories_discloses_uneven_domain_coverage(client):
    # HCG dispenses onco drugs through IP/OP BILLING, not internal consumption
    # movements — fact_consumption holds 3 rows / Rs 1,794 for the entire Onco Drugs
    # bucket against Rs 25.2 Cr of stock and Rs 297 Cr of billed revenue. So a user
    # picking "Onco Drugs" on a consumption-derived page legitimately sees ~nothing.
    # This must be DECLARED, or an empty chart reads as a broken filter.
    cats = {c["key"]: c for c in client.get("/meta/categories").json()["categories"]}

    onco = cats["Onco Drugs"]["coverage"]
    assert onco["has_stock"] is True
    assert onco["has_billing"] is True
    assert onco["has_consumption"] is False
    assert onco["note"] and "billing" in onco["note"].lower()
    assert onco["consumption_cost"] < 1e4 < onco["stock_value"]

    consumables = cats["Consumables"]["coverage"]
    assert consumables["has_consumption"] is True
    assert consumables["note"] is None


def test_unclassified_never_means_an_unknown_material():
    # Every material in every KPI table is present in dim_material, so
    # "Unclassified" only ever means "dim_material has no material_type for it" —
    # it is never a silent join failure.
    known = set(da.load("dim_material")["material"].astype(str))
    for table, col in (("kpi_stock_value", "material"), ("kpi_units_consumed", "material"),
                       ("stock_replenishment_and_aging_risk", "material_id")):
        missing = ~da.load(table)[col].astype(str).isin(known)
        assert int(missing.sum()) == 0, table


def test_endpoint_without_category_is_unchanged_and_with_category_subsets(client):
    plain = client.get("/kpi/current-stock-value/summary").json()
    onco = client.get("/kpi/current-stock-value/summary?Category=Onco Drugs").json()
    allcat = client.get("/kpi/current-stock-value/summary?Category=All").json()

    assert plain == allcat                                  # "All" == no filter
    assert plain["stock_value_cost"]["sum"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    assert onco["row_count"] < plain["row_count"]
    assert onco["stock_value_cost"]["sum"] == pytest.approx(251975495.01, abs=1.0)


def test_every_category_slice_of_an_endpoint_sums_back_to_the_whole(client):
    total = client.get("/kpi/current-stock-value/summary").json()["stock_value_cost"]["sum"]
    parts = 0.0
    for c in da.CATEGORIES:
        parts += client.get(f"/kpi/current-stock-value/summary?Category={c}") \
                       .json()["stock_value_cost"]["sum"]
    assert parts == pytest.approx(total, rel=1e-9)


def test_category_reaches_the_bespoke_insight_endpoints(client):
    whole = client.get("/kpi/inventory-valuation/insights").json()
    onco = client.get("/kpi/inventory-valuation/insights?Category=Onco Drugs").json()
    assert onco["totals"]["cost"] < whole["totals"]["cost"]
    assert onco["totals"]["cost"] == pytest.approx(251975495.01, abs=1.0)


def test_aging_distribution_insights_reproduces_the_committed_parquet_exactly(client):
    # The generic /kpi/aging-distribution keeps reading the committed parquet
    # untouched; this companion recomputes from fact_inventory so it CAN be
    # category-cut. Unfiltered, the two must agree to the rupee.
    body = client.get("/kpi/aging-distribution/insights").json()
    ad = da.load("kpi_aging_distribution")
    by = ad.groupby("aging_bucket", observed=True)["stock_value"].sum()
    for b in body["buckets"]:
        assert b["value"] == pytest.approx(float(by.get(b["bucket"], 0.0)), abs=0.01), b["bucket"]
    assert body["totals"]["total_value"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)

    onco = client.get("/kpi/aging-distribution/insights?Category=Onco Drugs").json()
    assert onco["totals"]["total_value"] == pytest.approx(251975495.01, abs=1.0)


def test_dashboard_all_category_scope_is_declared_honestly(client):
    body = client.get("/api/dashboard/all").json()
    assert body["categoryScope"]["applied"] is None
    # procurement / fill-rate source tables carry no material dimension, so the
    # response says so instead of pretending the selector reached them.
    assert "procurement" in body["categoryScope"]["unscoped"]

    onco = client.get("/api/dashboard/all?Category=Onco Drugs").json()
    assert onco["categoryScope"]["applied"] == "Onco Drugs"
    assert onco["stockValue"]["currentStockValue"] == pytest.approx(251975495.01, abs=1.0)
    # unscoped blocks are identical, exactly as declared
    assert onco["procurement"] == body["procurement"]
