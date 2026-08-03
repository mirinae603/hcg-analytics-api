# tests/test_drilldown.py — the generic hover drill-down.
#
# ONE endpoint behind every "hover a bar/slice -> show me the top 10 inside it"
# interaction, instead of a bespoke endpoint per chart. The properties that matter
# are arithmetic honesty, not cosmetics: the slice total must equal the bar the user
# hovered, shares must be OF THAT SLICE, and `covered_pct` must tell the reader how
# much of the bar the top-N actually accounts for — so a tooltip can never imply the
# ten rows it shows are the whole story when they are 12% of it.
from __future__ import annotations

import pytest

from app.core import data_access as da

TOTAL_STOCK_VALUE = 604679341.2
ONCO_STOCK_VALUE = 251975495.01


def test_drill_meta_advertises_the_contract(client):
    body = client.get("/drill/meta").json()
    assert body["default_n"] == 10
    for d in ("material_group", "plant", "vendor", "aging_band", "category"):
        assert d in body["dims"]
    assert "material" in body["by"]
    assert "current-stock-value" in body["kpis"]
    assert "reorder-priority" in body["kpis"]


def test_top_items_defaults_to_ten_and_ranks_descending(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=category"
                      "&slice=Onco Drugs").json()
    assert body["returned"] == 10
    vals = [i["value"] for i in body["items"]]
    assert vals == sorted(vals, reverse=True)
    assert [i["rank"] for i in body["items"]] == list(range(1, 11))
    assert all(i["name"] and i["key"] for i in body["items"])


def test_slice_total_equals_the_bar_the_user_hovered(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=category"
                      "&slice=Onco Drugs").json()
    assert body["slice_total"] == pytest.approx(ONCO_STOCK_VALUE, abs=1.0)
    assert body["grand_total"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    assert body["slice_share_pct"] == pytest.approx(41.67, abs=0.01)


def test_shares_are_of_the_slice_and_cumulative_is_consistent(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=category"
                      "&slice=Onco Drugs&n=10").json()
    items = body["items"]
    for i in items:
        assert i["share_pct"] == pytest.approx(i["value"] / body["slice_total"] * 100, abs=0.02)
    assert items[-1]["cum_share_pct"] == pytest.approx(body["covered_pct"], abs=0.02)
    assert 0 < body["covered_pct"] <= 100.0
    # the top-N is a genuine subset, and the response says how big the slice really is
    assert body["count"] > body["returned"]


def test_drill_by_material_group(client):
    whole = client.get("/drill/top-items?kpi=current-stock-value&dim=material_group"
                       "&by=material_group&n=5").json()
    slice_name = whole["items"][0]["key"]
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=material_group"
                      f"&slice={slice_name}").json()
    assert body["dim"] == "material_group"
    assert body["slice_total"] == pytest.approx(whole["items"][0]["value"], abs=1.0)


def test_drill_by_plant(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=plant&slice=AH01").json()
    sv = da.filter_plant(da.load("kpi_stock_value"), "AH01")
    assert body["slice_total"] == pytest.approx(float(sv["stock_value_cost"].sum()), abs=1.0)


def test_drill_by_vendor_dimension(client):
    body = client.get("/drill/top-items?kpi=vendor-volume-contribution&dim=plant"
                      "&slice=AH01&by=vendor&n=5").json()
    assert body["measure"] == "vendor_value"
    assert body["returned"] <= 5
    assert body["items"][0]["value"] > 0


def test_drill_by_derived_aging_band(client):
    body = client.get("/drill/top-items?kpi=inventory-aging&dim=aging_band&slice=365%2B").json()
    assert body["dim"] == "aging_band"
    ia = da.load("kpi_inventory_aging")
    expect = float(ia[ia["aging_days"] > 365]["closing_stock_value"].sum())
    assert body["slice_total"] == pytest.approx(expect, abs=1.0)


def test_drill_over_the_reorder_queue(client):
    body = client.get("/drill/top-items?kpi=reorder-priority&dim=category"
                      "&slice=Onco Drugs&measure=replenishment_quantity").json()
    assert body["measure"] == "replenishment_quantity"
    assert body["returned"] == 10
    assert body["items"][0]["value"] > 0


def test_slices_of_a_dimension_sum_back_to_the_grand_total(client):
    total = 0.0
    for c in da.CATEGORIES:
        body = client.get(f"/drill/top-items?kpi=current-stock-value&dim=category"
                          f"&slice={c}&n=1").json()
        total += body["slice_total"]
    assert total == pytest.approx(TOTAL_STOCK_VALUE, rel=1e-9)


def test_no_slice_means_the_whole_chart(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&n=10").json()
    assert body["slice"] is None
    assert body["slice_total"] == pytest.approx(TOTAL_STOCK_VALUE, abs=1.0)
    assert body["slice_share_pct"] == pytest.approx(100.0, abs=0.01)


def test_plant_and_category_filters_compose_with_the_drill(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=material_group"
                      "&Plant=AH01&Category=Onco Drugs&n=5").json()
    sv = da.filter_category(da.filter_plant(da.load("kpi_stock_value"), "AH01"), "Onco Drugs")
    assert body["grand_total"] == pytest.approx(float(sv["stock_value_cost"].sum()), abs=1.0)


def test_unknown_kpi_and_unsupported_dimension_are_rejected(client):
    assert client.get("/drill/top-items?kpi=not-a-kpi").status_code == 404
    r = client.get("/drill/top-items?kpi=vendor-lead-time&dim=aging_band")
    assert r.status_code == 400


def test_empty_slice_returns_nothing_rather_than_the_whole_chart(client):
    body = client.get("/drill/top-items?kpi=current-stock-value&dim=category"
                      "&slice=NoSuchCategory").json()
    assert body["returned"] == 0
    assert body["items"] == []
    assert body["slice_total"] == 0.0
    assert body["covered_pct"] == 0.0


# ── the three inventory charts wired last, each against ITS OWN endpoint ──────
# Every one of these drills goes through a source the chart does not itself read, so
# "200 with items" is not enough: the panel's slice_total has to equal the bar the
# user is pointing at, in every category, or the drill is confidently wrong.

def test_doh_band_drill_reproduces_the_coverage_histogram(client):
    """drill_doh_grain vs GET /kpi/days-on-hand/insights, band for band."""
    for cat in (None, "Consumables", "Other Drugs"):
        q = f"&Category={cat}" if cat else ""
        bands = client.get(f"/kpi/days-on-hand/insights?x=1{q}").json()["bands"]
        assert bands, "the histogram has no bands to compare against"
        for b in bands:
            d = client.get(f"/drill/top-items?kpi=days-on-hand&dim=doh_band&by=material"
                           f"&slice={b['key']}&measure=stock_value_cost&n=1{q}").json()
            assert d["source"] == "drill_doh_grain"
            assert d["slice_total"] == pytest.approx(b["value"], abs=0.01), \
                f"band {b['key']} (Category={cat}) drills to a different total than its bar"


def test_stock_change_drill_needs_the_year_in_the_slice(client):
    """dim=month would merge December 2025 into December 2026 under one bar."""
    months = client.get("/kpi/stock-change/insights").json()["monthly"]
    decembers = [m for m in months if m["month"] == "December"]
    assert len(decembers) == 2, "fixture no longer spans two Decembers — the trap is untested"

    for m in months:
        for meas in ("inflow", "outflow"):
            d = client.get(f"/drill/top-items?kpi=stock-change&dim=year_month&by=material"
                           f"&slice={m['year']}-{m['month']}&measure={meas}&n=1").json()
            assert d["slice_total"] == pytest.approx(m[meas], abs=0.01), \
                f"{m['label']} {meas} drills to a different total than its bar"

    # and the dimension that would have been the obvious choice really does over-report
    merged = client.get("/drill/top-items?kpi=stock-change&dim=month&by=material"
                        "&slice=December&measure=inflow&n=1").json()["slice_total"]
    assert merged == pytest.approx(sum(m["inflow"] for m in decembers), abs=0.01)
    assert merged > max(m["inflow"] for m in decembers)


def test_wastage_drill_ranks_the_numerator_not_the_ratio(client):
    """Per-plant expired value, equal to the bar's own expired_value in every category."""
    for cat in (None, "Onco Drugs", "Consumables"):
        q = f"&Category={cat}" if cat else ""
        w = client.get(f"/kpi/wastage-rate/insights?x=1{q}").json()
        for r in w["by_plant"]:
            d = client.get(f"/drill/top-items?kpi=wastage-rate&dim=plant&by=material"
                           f"&slice={r['plant']}&n=1{q}").json()
            assert d["measure"] == "expired_value"
            assert d["additive"] is True, "shares must decompose the rupees, never the %"
            assert d["slice_total"] == pytest.approx(r["expired_value"], abs=0.01), \
                f"plant {r['plant']} (Category={cat}) drills to a different total than its bar"
        # the whole chart's numerator is the page's own headline
        whole = client.get(f"/drill/top-items?kpi=wastage-rate&dim=plant&n=1{q}").json()
        assert whole["grand_total"] == pytest.approx(w["totals"]["expired_value"], abs=0.01)
