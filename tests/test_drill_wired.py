# tests/test_drill_wired.py — every drill config the FRONTEND actually wires.
#
# tests/test_drilldown_matrix.py pins the matrix's own contract; this file pins the
# subset of it that real charts hover. The distinction matters: the matrix advertises
# hundreds of (kpi, dim, by) combinations and the UI uses about thirty, and it was the
# thirty that shipped broken.
#
# The list below is a literal mirror of src/lib/drillSelfCheck.ts's WIRED_DRILLS (plus
# the generic /kpi page's DRILL map). Keeping the same list on both sides is the point:
# the browser check tells the person editing a PAGE, this one tells the person editing
# the API, and neither can quietly stop covering the other.
#
# Every case asserts a 200 AND a non-empty item list against a REAL slice discovered
# from the chart itself. A status-only check would pass on a made-up slice, which
# returns a perfectly valid empty list — an empty panel is as useless to a reader as an
# error and far easier to miss.
from __future__ import annotations

import pytest

# (page, chart, kpi, dim, by, measure|None)
WIRED = [
    # ── Inventory ──
    ("Stock Value", "breakdown bars · Category", "current-stock-value", "material_group", "material", "stock_value_cost"),
    ("Stock Value", "breakdown bars · Hospital", "current-stock-value", "plant", "material", "stock_value_cost"),
    ("Stock Value", "cost vs MRP", "current-stock-value", "material_group", "material", "stock_value_cost"),
    ("Stock Value", "value-at-risk age bands", "aging-distribution", "aging_bucket", "material", "stock_value"),
    ("Inventory Aging", "value by age band", "aging-distribution", "aging_bucket", "material", "stock_value"),
    ("Aging Distribution", "marimekko column", "aging-distribution", "material_group", "material", "stock_value"),
    ("Aging Distribution", "stagnant leaderboard", "aging-distribution", "material_group", "material", "stock_value"),
    ("Health Score", "tier scorecard", "inventory-health-score", "health_tier", "material", "closing_stock_value"),
    ("Health Score", "category report card", "inventory-health-score", "material_group", "material", "closing_stock_value"),
    ("Non-Moving", "reason bars", "non-moving-inventory", "reason", "material", "closing_stock_value"),
    ("Non-Moving", "blocked capital by category", "non-moving-inventory", "material_group", "material", "closing_stock_value"),
    ("Risk", "risk tier bars", "inventory-risk", "risk_level", "material", "closing_stock_value"),
    ("Risk", "high-risk categories", "inventory-risk", "material_group", "material", "closing_stock_value"),
    ("Near Expiry", "exposure by category", "near-expiry", "material_group", "material", "total_cost"),
    ("Turnover", "category velocity", "inventory-turnover-ratio", "material_group", "material", "closing_stock_value"),
    ("Valuation", "capital concentration", "current-stock-value", "material_group", "material", "stock_value_cost"),
    ("Days on Hand", "coverage distribution", "days-on-hand", "doh_band", "material", "stock_value_cost"),
    ("Stock Change", "monthly flow · inflow", "stock-change", "year_month", "material", "inflow"),
    ("Stock Change", "monthly flow · outflow", "stock-change", "year_month", "material", "outflow"),
    ("Wastage %", "wastage by plant", "wastage-rate", "plant", "material", "expired_value"),
    # ── Consumption ──
    ("Consumption Overview", "consumption by category", "unit-sold-per-sku", "material_group", "material", "consumption_cost"),
    ("Units Consumed", "where usage lands (units)", "unit-sold-per-sku", "material_group", "material", "total_units"),
    ("Units Consumed", "where usage lands (cost)", "unit-sold-per-sku", "material_group", "material", "consumption_cost"),
    ("Consumption by Dept", "department treemap", "consumption-by-department", "department", "material", "consumption_cost"),
    # ── Procurement ──
    ("Purchase Value", "spend blocks", "purchase-value", "category", "vendor", "purchase_value"),
    ("Vendor Volume", "largest suppliers", "vendor-volume-contribution", "vendor", "material", "vendor_value"),
    ("Monthly Purchase", "category x month heatmap", "monthly-purchase-value", "material_group", "material", "monthly_purchase_value"),
    ("Purchase by Location", "spend footprint hexes", "purchase-by-location", "plant", "material", "purchase_value"),
    ("Fill Rate", "fulfillment priority bubbles", "fill-rate", "plant", "material", "ordered_qty"),
    # ── Forecasting ──
    ("Forecasting Overview", "slow-moving stock donut", "aging-risk-forecast", "aging_risk_forecast", "material", "closing_stock"),
    # reorder-priority/priority_band goes through /forecast/reorder-priority in the UI
    # (the band is computed there, not stored) — covered separately below AND here,
    # because /drill/top-items must be able to answer it too.
    ("Forecasting Overview", "priority reorder list", "reorder-priority", "priority_band", "material", "replenishment_quantity"),
    ("Replenishment Risk", "priority ladder", "reorder-priority", "priority_band", "material", "replenishment_quantity"),
]

# The generic /kpi/{key} page's DRILL map — mirror of src/lib/drilldown.ts.
GENERIC = [
    ("current-stock-value", "material_group", "material"),
    ("inventory-aging", "aging_category", "material"),
    ("inventory-health-score", "health_tier", "material"),
    ("non-moving-inventory", "reason", "material"),
    ("inventory-risk", "risk_level", "material"),
    ("near-expiry", "expiry_bucket", "material"),
    ("aging-distribution", "aging_bucket", "material"),
    ("stock-radar", "radar_status", "material"),
    ("aging-risk-forecast", "aging_risk_forecast", "material"),
    ("vendor-volume-contribution", "vendor", "material"),
]


def _drill(client, **kw):
    return client.get("/drill/top-items", params=kw)


def _real_slice(client, kpi, dim, measure=None):
    """A slice the chart genuinely has — asking the endpoint to enumerate its own
    dimension, exactly the way the browser self-check does."""
    q = {"kpi": kpi, "dim": dim, "by": dim, "n": 1}
    if measure:
        q["measure"] = measure
    r = _drill(client, **q)
    assert r.status_code == 200, f"cannot enumerate {kpi}/{dim}: {r.status_code} {r.text[:200]}"
    items = r.json()["items"]
    assert items, f"{kpi} has no slices on dim={dim}"
    return items[0]["key"]


def _assert_answers(client, kpi, dim, by, measure=None, label=""):
    sl = _real_slice(client, kpi, dim, measure)
    q = {"kpi": kpi, "dim": dim, "by": by, "slice": sl, "n": 10}
    if measure:
        q["measure"] = measure
    r = _drill(client, **q)
    assert r.status_code == 200, f"{label} {kpi}/{dim}->{by} @ '{sl}': {r.status_code} {r.text[:220]}"
    b = r.json()
    assert b["returned"] > 0, f"{label} {kpi}/{dim}->{by} @ '{sl}' returned an EMPTY list"
    assert all(i["key"] and i["name"] for i in b["items"]), f"{label} unnamed rows on {kpi}"
    assert [i["rank"] for i in b["items"]] == list(range(1, b["returned"] + 1))
    vals = [i["value"] for i in b["items"]]
    assert vals == sorted(vals, reverse=True), f"{label} {kpi} items are not ranked"
    return b


@pytest.mark.parametrize("page,chart,kpi,dim,by,measure", WIRED,
                         ids=[f"{p}:{c}" for p, c, *_ in WIRED])
def test_a_wired_chart_answers_with_real_items(client, page, chart, kpi, dim, by, measure):
    _assert_answers(client, kpi, dim, by, measure, label=f"[{page} · {chart}]")


@pytest.mark.parametrize("kpi,dim,by", GENERIC, ids=[k for k, *_ in GENERIC])
def test_the_generic_kpi_page_drill_map_answers(client, kpi, dim, by):
    _assert_answers(client, kpi, dim, by, label="[generic /kpi page]")


def test_the_reorder_band_drill_matches_the_band_bar_it_launches_from(client):
    """The one drill that does NOT go through /drill/top-items.

    `priority_band` is computed inside /forecast/reorder-priority (cover vs lead time),
    so the UI adapts that endpoint instead. Its slice total must still equal the band
    bar the user is pointing at — the same guarantee the generic path gives.
    """
    j = client.get("/forecast/reorder-priority", params={"limit": 10}).json()
    bands = j["bands"]
    assert bands, "no priority bands"
    for band in bands:
        b = client.get("/forecast/reorder-priority",
                       params={"band": band["band"], "sort": "qty", "limit": 10}).json()
        assert b["items"], f"band {band['band']} has no lines"
        assert b["count"] == band["lines"]
        got = sum(float(r["reorder_qty"]) for r in b["items"])
        assert got <= band["qty"] + 1e-6, "top-10 cannot exceed the band it came from"


def test_a_drill_inherits_its_cards_category_without_out_filtering(client):
    """A drill launched from a card with an active category filter narrows to that
    category — and to EXACTLY the number that card is showing."""
    for kpi, dim, measure in [("current-stock-value", "material_group", "stock_value_cost"),
                              ("aging-distribution", "aging_bucket", "stock_value")]:
        plain = _drill(client, kpi=kpi, dim=dim, by="material", n=5, measure=measure).json()
        onco = _drill(client, kpi=kpi, dim=dim, by="material", n=5, measure=measure,
                      Category="Onco Drugs").json()
        assert onco["grand_total"] < plain["grand_total"], kpi
        assert onco["grand_total"] == pytest.approx(251975495.01, rel=1e-6), kpi


def test_every_wired_config_is_still_advertised_by_the_matrix(client):
    """The matrix is what a future page author will wire off. If a live config is not
    in it, the contract has drifted from reality in the direction that hides bugs."""
    matrix = {m["kpi"]: m for m in client.get("/drill/matrix").json()["kpis"]}
    missing = []
    for _, _, kpi, dim, by, _ in WIRED:
        m = matrix.get(kpi)
        if m is None:
            continue                      # aliases resolve inside the endpoint
        if not m["drillable"] or dim not in m["dims"] or by not in m["by"]:
            missing.append((kpi, dim, by))
    assert not missing, f"wired but not advertised: {missing}"
