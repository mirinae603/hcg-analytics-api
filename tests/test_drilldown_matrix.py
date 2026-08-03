# tests/test_drilldown_matrix.py — the drill-down's COVERAGE contract.
#
# tests/test_drilldown.py already pins the arithmetic of a single drill (shares are
# of the slice, covered_pct never overclaims). This file pins the thing that actually
# broke: six chart/dimension combinations the frontend had wired shipped returning
# 400, and because the frontend primitive swallows errors, a broken hover looked
# exactly like a chart nobody wired up. Nothing on screen said anything was wrong.
#
# So the properties asserted here are:
#   * every combination GET /drill/matrix advertises really answers 200 — the matrix
#     is the contract the frontend wires off, and a lying contract is the same bug
#     one layer up;
#   * the six known-broken combinations work, and work by RE-ROUTING to a table that
#     genuinely has the grain rather than by inventing numbers — each is checked
#     against the pre-aggregated table it must reproduce;
#   * a genuinely impossible combination still fails CLEANLY (400 with a reason a
#     human can act on), never 500, and never a fabricated answer.
from __future__ import annotations

import pytest

from app.core import data_access as da

TOTAL_STOCK_VALUE = 604679341.2


def drill(client, **kw):
    return client.get("/drill/top-items", params=kw)


def body(client, **kw):
    r = drill(client, **kw)
    assert r.status_code == 200, f"{kw} -> {r.status_code} {r.text[:200]}"
    return r.json()


# ── the six that shipped broken ──────────────────────────────────────────────
# Each is asserted against the number the CHART shows, not merely for a 200: a
# re-routed drill that returns a plausible-looking list of the wrong rows would be a
# worse failure than the 400 it replaces.

def test_aging_distribution_by_bucket_now_returns_materials(client):
    b = body(client, kpi="aging-distribution", dim="aging_bucket", slice="0-30", by="material")
    ad = da.load("kpi_aging_distribution")
    expect = float(ad[ad["aging_bucket"].astype(str) == "0-30"]["stock_value"].sum())
    # re-routed to the fact grain, and still equal to the pre-aggregated bar
    assert b["rerouted"] is True
    assert b["source"] == "drill_inventory_grain"
    assert b["slice_total"] == pytest.approx(expect, rel=1e-9)
    assert b["measure"] == "stock_value"
    assert b["returned"] == 10
    assert all(i["value"] > 0 and i["name"] for i in b["items"])


def test_aging_distribution_by_material_group_now_returns_materials(client):
    ad = da.load("kpi_aging_distribution")
    grp = str(ad.groupby("material_group", observed=True)["stock_value"].sum().idxmax())
    b = body(client, kpi="aging-distribution", dim="material_group", slice=grp, by="material")
    expect = float(ad[ad["material_group"].astype(str) == grp]["stock_value"].sum())
    assert b["slice_total"] == pytest.approx(expect, rel=1e-9)
    assert b["returned"] > 0


def test_purchase_value_by_spend_category_now_returns_materials(client):
    pv = da.load("kpi_purchase_value")
    cat = str(pv.groupby("category", observed=True)["purchase_value"].sum().idxmax())
    b = body(client, kpi="purchase-value", dim="category", slice=cat, by="material")
    expect = float(pv[pv["category"].astype(str) == cat]["purchase_value"].sum())
    assert b["source"] == "drill_po_grain"
    assert b["measure"] == "purchase_value"
    assert b["slice_total"] == pytest.approx(expect, rel=1e-9)
    assert b["returned"] > 0


def test_vendor_volume_by_vendor_now_returns_materials(client):
    vv = da.load("kpi_vendor_volume")
    ven = str(vv.groupby("vendor_name", observed=True)["vendor_value"].sum().idxmax())
    b = body(client, kpi="vendor-volume-contribution", dim="vendor", slice=ven, by="material")
    expect = float(vv[vv["vendor_name"].astype(str) == ven]["vendor_value"].sum())
    assert b["source"] == "drill_po_grain"
    # the parent chart's own measure NAME survives the re-route
    assert b["measure"] == "vendor_value"
    assert b["slice_total"] == pytest.approx(expect, rel=1e-9)
    assert b["returned"] > 0


def test_reorder_priority_supports_its_own_priority_band(client):
    b = body(client, kpi="reorder-priority", dim="priority_band", slice="1",
             by="material", measure="replenishment_quantity")
    assert b["dim"] == "priority_band"
    assert b["source"] == "drill_replen_priority"
    assert b["returned"] == 10
    # …and agrees with the band bar the user is hovering, which is drawn from
    # /forecast/reorder-priority's own band summary
    bands = client.get("/forecast/reorder-priority").json()["bands"]
    band1 = next(x for x in bands if x["band"] == 1)
    assert b["slice_total"] == pytest.approx(band1["qty"], rel=1e-9)


def test_inventory_valuation_is_registered(client):
    b = body(client, kpi="inventory-valuation", dim="material_group", by="material")
    assert b["kpi_resolved"] == "current-stock-value"
    assert b["grand_total"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    assert b["returned"] == 10


# ── the matrix is the contract: everything it advertises must answer ──────────

def _matrix(client):
    """Only the KPIs the matrix says are drillable.

    A KPI with no entity grain anywhere in its chain (forecast_accuracy is six
    (metric, value) rows) now advertises NOTHING — `drillable: false` and an empty
    `dims` — because a contract that advertises a dimension the endpoint then 400s on
    is the same "broken looks identical to unwired" bug one layer up. The
    non-drillable case has its own test below.
    """
    return [m for m in client.get("/drill/matrix").json()["kpis"] if m["drillable"]]


def test_a_kpi_with_no_entity_grain_advertises_nothing_and_says_why(client):
    m = next(k for k in client.get("/drill/matrix").json()["kpis"]
             if k["kpi"] == "forecast-accuracy")
    assert m["drillable"] is False
    assert m["dims"] == {}                       # nothing to wire a dead hover off
    assert "no material" in m["not_drillable_reason"]
    # and the endpoint agrees with its own contract
    assert drill(client, kpi="forecast-accuracy", dim="metric", by="material").status_code == 400


def test_every_drillable_kpi_is_reachable_with_by_material(client):
    """The default `by` on every drillable KPI must answer, because that is what a
    caller who names only a kpi + dim gets."""
    for m in _matrix(client):
        body(client, kpi=m["kpi"], by="material", n=1)


def test_every_advertised_dim_and_by_combination_answers_200(client):
    broken = []
    for m in _matrix(client):
        for dim in m["dims"]:
            for by in m["by"]:
                r = drill(client, kpi=m["kpi"], dim=dim, by=by, n=5)
                if r.status_code != 200:
                    broken.append((m["kpi"], dim, by, r.status_code, r.text[:120]))
    assert not broken, f"{len(broken)} advertised combinations do not answer: {broken[:10]}"


def test_every_advertised_combination_returns_a_sane_top_n(client):
    thin = []
    for m in _matrix(client):
        for dim in m["dims"]:
            b = body(client, kpi=m["kpi"], dim=dim, by="material", n=10)
            if not (0 < b["returned"] <= 10):
                thin.append((m["kpi"], dim, b["returned"]))
            ranks = [i["rank"] for i in b["items"]]
            assert ranks == list(range(1, len(ranks) + 1))
            vals = [i["value"] for i in b["items"]]
            assert vals == sorted(vals, reverse=True)
            assert all(i["key"] and i["name"] for i in b["items"])
    assert not thin, f"combinations returning nothing: {thin}"


def test_matrix_names_a_real_source_and_measure_for_every_kpi(client):
    for m in _matrix(client):
        assert m["sources"] and m["primary_table"]
        assert m["default_measure"], m["kpi"]
        b = body(client, kpi=m["kpi"], n=1)
        assert b["measure"] == m["default_measure"]
        assert b["additive"] is m["measure_additive"]


# ── failure is still clean and still says why ────────────────────────────────

def test_genuinely_unsupported_combinations_are_a_clean_400(client):
    # lead time is per-vendor and carries no aging at all — no source can invent one
    r = drill(client, kpi="vendor-lead-time", dim="aging_band")
    assert r.status_code == 400
    assert "aging_band" in r.json()["detail"]
    assert "/drill/matrix" in r.json()["detail"]     # the reply says where to look

    # forecast_accuracy is six (metric, value) rows — there is no entity to drill into
    r = drill(client, kpi="forecast-accuracy", dim="metric", by="material")
    assert r.status_code == 400
    assert "no entity grain" in r.json()["detail"]

    # and an unknown key is still a 404, not a 400 and not a 500
    assert drill(client, kpi="not-a-kpi").status_code == 404


def test_no_supported_combination_ever_500s(client):
    for m in _matrix(client):
        for dim in list(m["dims"]) + ["definitely_not_a_dimension"]:
            for by in list(m["by"]) + ["definitely_not_a_grain"]:
                r = drill(client, kpi=m["kpi"], dim=dim, by=by, slice="___nothing___", n=3)
                assert r.status_code in (200, 400), (m["kpi"], dim, by, r.status_code, r.text[:200])


# ── degradation is reported, never silent ────────────────────────────────────

def test_unavailable_by_degrades_and_says_so(client):
    # the replenishment queue has no vendor dimension anywhere
    b = body(client, kpi="reorder-priority", dim="priority_band", slice="1", by="vendor")
    assert b["by_fallback"] is True
    assert b["by_requested"] == "vendor"
    assert b["by_actual"] != "vendor"
    assert b["note"] and "vendor" in b["note"]
    assert b["returned"] > 0


def test_a_served_by_is_never_marked_as_a_fallback(client):
    b = body(client, kpi="current-stock-value", dim="material_group", by="material")
    assert b["by_fallback"] is False
    assert b["by_actual"] == "material" and b["note"] is None
    assert b["rerouted"] is False


# ── non-additive measures are aggregated correctly, not summed ───────────────

def test_ratio_measures_are_weighted_means_not_sums(client):
    b = body(client, kpi="vendor-lead-time", dim="vendor", by="material",
             measure="avg_lead_time_days")
    assert b["additive"] is False
    assert b["agg"] == "weighted_mean" and b["weight"]
    assert b["items"][0]["share_pct"] is None       # a % of an average is meaningless
    assert b["covered_pct"] is None
    assert b["note"] and "not additive" in b["note"]
    # a lead time is days, not a running total of days
    assert 0 <= b["items"][0]["value"] <= 365
    assert 0 <= b["grand_total"] <= 365


def test_fill_rate_weighted_mean_matches_the_plant_aggregate(client):
    fr = da.load("kpi_fill_rate")
    b = body(client, kpi="fill-rate", dim="plant", by="material", measure="fill_rate_pct")
    expect = float((1 - fr["open_qty"].sum() / fr["ordered_qty"].sum()) * 100)
    assert b["grand_total"] == pytest.approx(expect, rel=1e-6)
    assert b["additive"] is False


# ── nothing is silently dropped, and Plant + Category still compose ──────────

def test_no_slice_means_the_whole_chart_on_every_kpi(client):
    """slice_total with no slice must equal grand_total — i.e. the group-by threw
    nothing away. This is the regression that hid Rs 217 Cr of purchase value behind
    a null material_desc."""
    off = []
    for m in _matrix(client):
        b = body(client, kpi=m["kpi"], by="material", n=1)
        if b["additive"] and b["grand_total"]:
            if abs(b["slice_total"] - b["grand_total"]) / abs(b["grand_total"]) > 1e-9:
                off.append((m["kpi"], b["slice_total"], b["grand_total"]))
    assert not off, f"rows dropped by the group-by: {off}"


def test_rows_with_no_key_are_labelled_not_dropped(client):
    b = body(client, kpi="monthly-purchase-value", dim="material_group", by="material", n=10)
    mpv = da.load("kpi_monthly_purchase_value")
    assert b["grand_total"] == pytest.approx(float(mpv["monthly_purchase_value"].sum()), rel=1e-9)
    assert b["slice_total"] == pytest.approx(b["grand_total"], rel=1e-9)
    assert not any(i["name"] == "nan" for i in b["items"])


def test_plant_and_category_compose_on_a_rerouted_drill(client):
    a = body(client, kpi="aging-distribution", dim="aging_bucket", slice="0-30",
             by="material", Plant="AH01")
    ad = da.filter_plant(da.load("kpi_aging_distribution"), "AH01")
    expect = float(ad[ad["aging_bucket"].astype(str) == "0-30"]["stock_value"].sum())
    assert a["slice_total"] == pytest.approx(expect, rel=1e-9)

    c = body(client, kpi="current-stock-value", dim="aging_bucket", by="material",
             Plant="AH01", Category="Onco Drugs")
    sv = da.filter_category(da.filter_plant(da.load("kpi_stock_value"), "AH01"), "Onco Drugs")
    assert c["grand_total"] == pytest.approx(float(sv["stock_value_cost"].sum()), rel=1e-6)


def test_a_rerouted_drill_never_out_filters_the_chart_it_came_from(client):
    """A drill matches its chart's category behaviour EXACTLY — in both directions.

    Stated as the invariant rather than as one example, because the example keeps
    moving. kpi_aging_distribution was once in the "ignores it" group and is not any
    more; kpi_purchase_value has now followed it, because /kpi/purchase-value is served
    from fact_po at material grain whenever a category is passed. A drill still
    answering Rs 649.9 Cr under a Rs 236.7 Cr bar would be exactly the lie this test
    exists to catch, just inverted — so the assertion is "the drill total equals the
    chart total", never "the drill ignores the filter".
    """
    # honours it — the drill narrows to the same total the CHART narrows to, and the
    # `category` dimension it is grouped by stays the PO SPEND taxonomy throughout.
    plain = body(client, kpi="purchase-value", dim="category", by="vendor")
    onco = body(client, kpi="purchase-value", dim="category", by="vendor", Category="Onco Drugs")
    assert onco["grand_total"] < plain["grand_total"]
    chart = client.get("/kpi/purchase-value/insights",
                       params={"Category": "Onco Drugs"}).json()
    assert onco["grand_total"] == pytest.approx(chart["totals"]["spend"], rel=1e-6)

    # ...and the one procurement KPI that refuses the cut refuses it on the drill too,
    # rather than quietly answering with the whole portfolio.
    r = client.get("/drill/top-items?kpi=vendor-lead-time&by=vendor&Category=Onco%20Drugs")
    assert r.status_code == 400

    # honours it — and the drill narrows to the same total the chart shows
    a = body(client, kpi="aging-distribution", dim="aging_bucket", slice="0-30", by="material")
    b = body(client, kpi="aging-distribution", dim="aging_bucket", slice="0-30",
             by="material", Category="Onco Drugs")
    assert b["slice_total"] < a["slice_total"]
    chart = client.get("/kpi/aging-distribution", params={
        "group_by": "aging_bucket", "measures": "stock_value", "Category": "Onco Drugs"}).json()
    bar = next(r for r in chart if r["aging_bucket"] == "0-30")
    assert b["slice_total"] == pytest.approx(bar["stock_value"], rel=1e-9)

    c = body(client, kpi="current-stock-value", dim="material_group", by="material")
    d = body(client, kpi="current-stock-value", dim="material_group", by="material",
             Category="Onco Drugs")
    assert d["grand_total"] < c["grand_total"]
