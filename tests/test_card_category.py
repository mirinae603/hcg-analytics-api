# tests/test_card_category.py — the CARD-LEVEL category filter's contract.
#
# The category selector used to live in the page header, which meant it appeared above
# charts it could not narrow. It now lives on the individual card, and its PRESENCE is
# the claim: "this card can be split by material category". A card wired to an endpoint
# that silently ignores `?Category=` would be exactly the old bug in a smaller box —
# an inert control — so this file pins BOTH halves of that claim:
#
#   * every endpoint a card's filter is wired to really moves under `?Category=`, and
#     moves to the RIGHT number (checked against /meta/categories' own per-bucket
#     totals, not merely "different");
#   * every endpoint deliberately left WITHOUT a filter really is inert, so if a future
#     change gives one of them material grain, this file fails and tells someone to go
#     wire the card rather than leaving the capability undiscovered.
#
# The `?Category=` param itself is older than the card filter and already covered by
# tests/test_category_filter.py; what is new here is the wiring contract.
from __future__ import annotations

import pytest

from app.core import data_access as da

TOTAL_STOCK_VALUE = 604679341.2
ONCO_STOCK_VALUE = 251975495.01
CONSUMABLES_STOCK_VALUE = 220353104.76

# ── endpoints that a card's category chip is wired to ────────────────────────
# (path, extra query params). Each must return DIFFERENT json under a category.
WIRED = [
    ("/portfolio/inventory/summary", {}),
    ("/kpi/current-stock-value/summary", {}),
    ("/kpi/current-stock-value", {"group_by": "material_group", "measures": "stock_value_cost"}),
    ("/kpi/inventory-aging", {}),
    ("/kpi/inventory-aging/summary", {}),
    ("/kpi/aging-distribution", {"group_by": "aging_bucket", "measures": "stock_value"}),
    ("/kpi/aging-distribution/insights", {}),
    ("/kpi/days-on-hand/insights", {}),
    ("/kpi/inventory-health-score/insights", {}),
    ("/kpi/non-moving-inventory/insights", {}),
    ("/kpi/inventory-risk/insights", {}),
    ("/kpi/near-expiry/insights", {}),
    ("/kpi/near-expiry/items", {"limit": "25"}),
    ("/kpi/near-expiry", {}),
    ("/kpi/wastage-rate/insights", {}),
    ("/kpi/inventory-turnover-ratio/insights", {}),
    ("/kpi/inventory-turnover-ratio", {}),
    ("/kpi/inventory-valuation/insights", {}),
    ("/kpi/inventory-valuation", {}),
    ("/kpi/stock-change/insights", {}),
    ("/kpi/stock-change", {}),
    ("/portfolio/consumption/overview", {}),
    ("/kpi/unit-sold-per-sku/insights", {}),
    ("/kpi/unit-sold-per-sku", {}),
    ("/portfolio/forecasting/overview", {}),
    ("/forecast/demand-insights", {}),
    ("/forecast/cashflow-insights", {}),
    ("/forecast/replenishment-insights", {}),
    ("/forecast/reorder-priority", {"limit": "10"}),
    ("/forecast/risk-items", {"limit": "10"}),
    ("/inventory/replenishment-data", {}),
    ("/kpi/monthly-purchase-value/insights", {}),
    ("/kpi/monthly-purchase-value", {}),
]

# ── endpoints deliberately left WITHOUT a card filter, and why ───────────────
# Every one of these was curled with and without the param and came back identical.
# The reason is always the same shape: the aggregate was cut past material grain, so
# there is no category column to filter on and inventing one would be a lie.
NOT_WIRED = [
    ("/portfolio/procurement/overview", {}),        # spend/vendor/cycle aggregates, no material
    ("/portfolio/procurement/savings", {}),         # GRN price-variance rollup, no material
    ("/kpi/purchase-value/insights", {}),           # kpi_purchase_value: `category` is the PO
    ("/kpi/procurement-variance/insights", {}),     #   SPEND taxonomy (1,360 values), not ours
    ("/kpi/vendor-volume-contribution/insights", {}),   # plant x vendor
    ("/kpi/purchase-by-location/insights", {}),     # one row per plant
    ("/kpi/procurement-cycle-time/insights", {}),   # plant x month
    ("/kpi/vendor-lead-time/insights", {}),         # one row per vendor
    ("/kpi/fill-rate/insights", {}),                # one row per plant
    ("/kpi/vendor-volume-vs-margin", {}),           # GRN rollup to vendor
    ("/kpi/consumption-by-department/insights", {}),    # plant x cost centre x month
    ("/revenue/insights", {}),                      # billing aggregates keyed on their own
    ("/revenue/items", {"limit": "10"}),            #   `group`, not the derived category
]

CATS = ["Onco Drugs", "Consumables"]


def _get(client, path, params):
    r = client.get(path, params=params)
    assert r.status_code == 200, f"{path} {params} -> {r.status_code} {r.text[:200]}"
    return r.json()


@pytest.mark.parametrize("path,extra", WIRED, ids=[p for p, _ in WIRED])
def test_a_wired_card_endpoint_really_moves_under_category(client, path, extra):
    base = _get(client, path, extra)
    for c in CATS:
        got = _get(client, path, {**extra, "Category": c})
        assert got != base, f"{path} ignores ?Category={c} — the card's chip would be inert"


@pytest.mark.parametrize("path,extra", NOT_WIRED, ids=[p for p, _ in NOT_WIRED])
def test_an_unwired_card_endpoint_really_is_inert(client, path, extra):
    """If this fails the endpoint GAINED material grain — go wire the card, don't
    delete the case."""
    base = _get(client, path, extra)
    for c in CATS:
        assert _get(client, path, {**extra, "Category": c}) == base, (
            f"{path} now honours ?Category={c} — its card should have a filter")


# ── the regrain: kpi_aging_distribution has no material column at all ─────────

def test_aging_distribution_regrain_reproduces_the_parquet_cell_for_cell(client):
    """The substitute frame is not a lookalike — it IS the parquet, un-collapsed.

    kpi_aging_distribution is plant x material_group x aging_bucket, so `?Category=`
    had nothing to filter and came back identical under every bucket: an invisible
    no-op. It is now served from drill_inventory_grain whenever (and only when) a
    category is passed, which is only legitimate if collapsing that frame back onto
    the parquet's own grain reproduces it exactly.
    """
    from app.core import drill_sources as ds

    par = da.load("kpi_aging_distribution")
    gr = ds.inventory_grain()
    keys = ["plant", "material_group", "aging_bucket"]
    meas = ["stock_value", "stock_qty", "sku_count"]
    a = par.groupby(keys, observed=True)[meas].sum().reset_index()
    b = gr.groupby(keys, observed=True)[meas].sum().reset_index()
    m = a.merge(b, on=keys, how="outer", suffixes=("_p", "_g")).fillna(0)
    assert len(m) == len(a) == len(b)
    for c in meas:
        assert (m[f"{c}_p"] - m[f"{c}_g"]).abs().max() == pytest.approx(0.0, abs=1e-6), c


def test_aging_distribution_is_untouched_without_a_category(client):
    """No category => the original code path against the original parquet, so the
    unfiltered chart is not merely equal, it is the same computation."""
    par = da.load("kpi_aging_distribution")
    rows = _get(client, "/kpi/aging-distribution",
                {"group_by": "aging_bucket", "measures": "stock_value,stock_qty,sku_count"})
    assert sum(r["stock_value"] for r in rows) == pytest.approx(TOTAL_STOCK_VALUE, rel=1e-9)
    for r in rows:
        exp = par[par["aging_bucket"].astype(str) == r["aging_bucket"]]
        assert r["stock_value"] == pytest.approx(float(exp["stock_value"].sum()), rel=1e-9)
        assert r["sku_count"] == pytest.approx(float(exp["sku_count"].sum()), rel=1e-9)


@pytest.mark.parametrize("cat,total", [("Onco Drugs", ONCO_STOCK_VALUE),
                                       ("Consumables", CONSUMABLES_STOCK_VALUE)])
def test_aging_distribution_under_a_category_totals_that_category(client, cat, total):
    rows = _get(client, "/kpi/aging-distribution",
                {"group_by": "aging_bucket", "measures": "stock_value", "Category": cat})
    assert sum(r["stock_value"] for r in rows) == pytest.approx(total, rel=1e-6)


def test_aging_distribution_buckets_sum_to_the_portfolio_across_all_categories(client):
    """Every bucket of every category, added up, is still the whole portfolio — the
    regrain partitions the stock, it does not duplicate or drop any of it."""
    cats = [c["key"] for c in _get(client, "/meta/categories", {})["categories"] if c["key"] != "All"]
    total = 0.0
    for c in cats:
        rows = _get(client, "/kpi/aging-distribution",
                    {"group_by": "aging_bucket", "measures": "stock_value", "Category": c})
        total += sum(r["stock_value"] for r in rows)
    assert total == pytest.approx(TOTAL_STOCK_VALUE, rel=1e-6)


def test_the_aging_distribution_drill_agrees_with_its_own_filtered_chart(client):
    """The panel must never print a bigger total than the bar it was launched from.

    The drill's first source is the pre-aggregated parquet, which has no category
    column — serving the drill from it under a category would have over-reported. The
    resolver now skips a source that cannot honour an active category.
    """
    for cat in (None, "Onco Drugs"):
        q = {"group_by": "aging_bucket", "measures": "stock_value"}
        if cat:
            q["Category"] = cat
        rows = _get(client, "/kpi/aging-distribution", q)
        bar = next(r for r in rows if r["aging_bucket"] == "0-30")
        d = {"kpi": "aging-distribution", "dim": "aging_bucket", "slice": "0-30", "by": "material"}
        if cat:
            d["Category"] = cat
        got = _get(client, "/drill/top-items", d)
        assert got["source"] == "drill_inventory_grain"
        assert got["slice_total"] == pytest.approx(bar["stock_value"], rel=1e-9)
        assert got["grand_total"] == pytest.approx(sum(r["stock_value"] for r in rows), rel=1e-9)
        assert got["returned"] > 0


# ── monthly purchase value: the one procurement table with material grain ────

def test_monthly_purchase_value_is_the_only_procurement_money_metric_with_grain(client):
    """kpi_monthly_purchase_value kept its material column, so it — alone among the
    procurement aggregates — can honestly be cut by material category."""
    mpv = da.load("kpi_monthly_purchase_value")
    total = float(mpv["monthly_purchase_value"].sum())
    base = _get(client, "/kpi/monthly-purchase-value/insights", {})
    assert base["totals"]["total"] == pytest.approx(total, rel=1e-9)

    seen = 0.0
    for c in [x["key"] for x in _get(client, "/meta/categories", {})["categories"] if x["key"] != "All"]:
        got = _get(client, "/kpi/monthly-purchase-value/insights", {"Category": c})
        exp = float(mpv[mpv["category"].astype(str) == c]["monthly_purchase_value"].sum())
        assert got["totals"]["total"] == pytest.approx(exp, rel=1e-9), c
        seen += got["totals"]["total"]
    assert seen == pytest.approx(total, rel=1e-9)


def test_purchase_value_still_refuses_to_be_cut_by_material_category(client):
    """kpi_purchase_value's own `category` is the PO SPEND taxonomy, so filtering it
    with a material bucket must stay a no-op rather than silently returning Rs 0 —
    which is what made the two procurement money metrics contradict each other."""
    base = _get(client, "/kpi/purchase-value/summary", {})
    for c in CATS:
        assert _get(client, "/kpi/purchase-value/summary", {"Category": c}) == base


# ── the option list the chip renders ─────────────────────────────────────────

def test_meta_categories_carries_what_the_chip_needs(client):
    j = _get(client, "/meta/categories", {})
    cats = j["categories"]
    assert cats[0]["key"] == "All"
    keys = [c["key"] for c in cats]
    assert "Onco Drugs" in keys and "Unclassified" in keys
    onco = next(c for c in cats if c["key"] == "Onco Drugs")
    assert onco["value"] == pytest.approx(ONCO_STOCK_VALUE, rel=1e-6)
    # the chip greys a bucket out per DOMAIN, and explains an empty card with the
    # backend's own wording — both need these fields present on every bucket
    for c in cats[1:]:
        cov = c["coverage"]
        for k in ("has_stock", "has_consumption", "has_billing", "has_reorder"):
            assert isinstance(cov[k], bool), (c["key"], k)
    # onco is the case the note exists for: real stock, no internal consumption
    assert onco["coverage"]["has_stock"] is True
    assert onco["coverage"]["has_consumption"] is False
    assert onco["coverage"]["note"]
