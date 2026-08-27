# tests/test_reorder_priority.py — the reorder-list COVERAGE defect.
#
# The bug: `status` in legacy_kpi._replen_frame is a first-match-wins if-chain, so
# every row gets exactly ONE of Stock-out / Dead stock / Reorder now / Overstocked /
# Healthy. "Needs replenishing" is an independent PROPERTY, not one of those five
# states — so pairing it with the label buried 17,907 of the 19,014 lines that
# genuinely need ordering, including all 15,878 that are ALREADY AT ZERO STOCK.
#
# These tests pin the fix from both ends: the queue must be COMPLETE (all 19,014),
# correctly PRIORITISED (zero-stock first), and the old display status must still be
# there for context — while every pre-existing field keeps its original meaning.
from __future__ import annotations

import pytest

from app.api import legacy_kpi as lk
from app.core import data_access as da

# Verified directly off stock_replenishment_and_aging_risk.parquet (67,167 rows).
TOTAL_LINES = 67167
NEEDS_REORDER = 19014
LABELLED_REORDER_NOW = 1107          # what the old view called "the reorder list"
STOCKOUT_LINES = 15878               # the most urgent lines, previously excluded
PRICED_LINES = 3129                  # only ~16% carry a unit cost
PRICED_VALUE = 72056217.71           # Rs 7.2056 Cr over the priced subset
STATUS_TOTALS = {"Overstocked": 38561, "Stock-out": 15878, "Dead stock": 8388,
                 "Healthy": 3233, "Reorder now": 1107}
HIDDEN_UNDER = {"Stock-out": 15878, "Healthy": 1700, "Reorder now": 1107,
                "Dead stock": 165, "Overstocked": 164}


@pytest.fixture(scope="module")
def rp():
    return lk._replen_frame(None)


# ---------- the defect is real, and `status` is still exactly what it was ----------

def test_display_status_is_unchanged(rp):
    assert len(rp) == TOTAL_LINES
    assert rp["status"].value_counts().to_dict() == STATUS_TOTALS


def test_the_coverage_gap_the_fix_exists_to_close(rp):
    need = rp[rp["replenishment_quantity"] > 0]
    assert len(need) == NEEDS_REORDER
    assert need["status"].value_counts().to_dict() == HIDDEN_UNDER
    # only 5.8% of the real reorder list was ever labelled "Reorder now"
    assert LABELLED_REORDER_NOW / NEEDS_REORDER < 0.06


# ---------- the priority dimension is orthogonal, not a replacement ----------

def test_priority_band_is_orthogonal_to_display_status(rp):
    need = rp[rp["needs_reorder"]]
    # a line can be BOTH "Stock-out" for display AND band 1 of the queue
    stockout = need[need["status"] == "Stock-out"]
    assert len(stockout) == STOCKOUT_LINES
    assert set(stockout["priority_band"].unique()) == {1}
    # lines that need nothing get band 0, so a well-stocked line can never be
    # mistaken for a low-priority order
    assert set(rp[~rp["needs_reorder"]]["priority_band"].unique()) == {0}


def test_every_line_needing_replenishment_is_in_the_queue(rp):
    need = lk._priority_sorted(rp)
    assert len(need) == NEEDS_REORDER
    assert (need["replenishment_quantity"] > 0).all()
    assert set(need["priority_band"].unique()) <= {1, 2, 3, 4, 5}


def test_queue_is_ordered_most_urgent_first(rp):
    need = lk._priority_sorted(rp)
    bands = need["priority_band"].tolist()
    assert bands == sorted(bands), "bands must be monotonically non-decreasing"
    assert bands[0] == 1
    assert need["priority_rank"].tolist() == list(range(1, len(need) + 1))
    # an item already at zero stock outranks anything that still has cover
    assert (need[need["priority_band"] == 1]["closing_stock"] <= 0).all()
    first_non_band1 = need[need["priority_band"] > 1].iloc[0]
    assert first_non_band1["priority_rank"] > STOCKOUT_LINES


def test_within_a_band_higher_demand_comes_first(rp):
    need = lk._priority_sorted(rp)
    for b in (1, 2, 3):
        seg = need[need["priority_band"] == b]["demand_monthly"].tolist()
        assert seg == sorted(seg, reverse=True), f"band {b} not demand-ordered"


def test_band_boundaries_follow_cover(rp):
    need = lk._priority_sorted(rp)
    assert (need[need["priority_band"] == 2]["cover"] < 0.5).all()
    assert (need[need["priority_band"] == 3]["cover"] < 1.0).all()
    assert (need[need["priority_band"] == 4]["cover"] < 3.0).all()
    assert (need[need["priority_band"] == 5]["cover"] >= 3.0).all()


def test_bands_partition_the_queue_with_no_line_lost(rp):
    need = lk._priority_sorted(rp)
    bands = lk._priority_bands_summary(need)
    assert [b["band"] for b in bands] == [1, 2, 3, 4, 5]
    assert sum(b["lines"] for b in bands) == NEEDS_REORDER
    assert bands[0]["lines"] == STOCKOUT_LINES
    assert bands[0]["label"] == "Out of stock"


# ---------- the pricing-coverage disclosure is never dropped ----------

def test_pricing_gap_is_disclosed_not_hidden(rp):
    need = lk._priority_sorted(rp)
    t = lk._reorder_totals(rp, need)
    assert t["reorder_lines"] == NEEDS_REORDER
    assert t["priced_lines"] == PRICED_LINES
    assert t["unpriced_lines"] == NEEDS_REORDER - PRICED_LINES
    assert t["reorder_value_priced"] == pytest.approx(PRICED_VALUE, abs=1.0)
    assert t["priced_share_pct"] == pytest.approx(16.5, abs=0.1)
    assert t["out_of_stock_lines"] == STOCKOUT_LINES
    # Thousands-separated: the disclosure is one long sentence carrying several 5-digit
    # counts, so "19,014" reads where "19014" does not. Intent is unchanged — the caveat
    # must still state the real priced share and the real total.
    assert "16.5%" in t["value_disclosure"] and f"{NEEDS_REORDER:,}" in t["value_disclosure"]


# ---------- the endpoint ----------

def test_reorder_priority_endpoint_covers_everything(client):
    r = client.get("/forecast/reorder-priority?limit=25")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == NEEDS_REORDER
    assert body["totals"]["reorder_lines"] == NEEDS_REORDER
    assert body["totals"]["priced_lines"] == PRICED_LINES
    assert sum(b["lines"] for b in body["bands"]) == NEEDS_REORDER


def test_reorder_priority_endpoint_puts_zero_stock_at_the_top(client):
    items = client.get("/forecast/reorder-priority?limit=25").json()["items"]
    assert items[0]["priority_rank"] == 1
    assert all(i["priority_band"] == 1 and i["stock"] <= 0 for i in items)
    assert [i["priority_rank"] for i in items] == list(range(1, 26))
    # the display status rides along for context rather than being destroyed
    assert all(i["status"] == "Stock-out" for i in items)
    assert all(i["reorder_qty"] > 0 for i in items)


def test_reorder_priority_paginates_and_filters_by_band(client):
    p2 = client.get("/forecast/reorder-priority?limit=10&offset=10").json()
    assert [i["priority_rank"] for i in p2["items"]] == list(range(11, 21))
    b3 = client.get("/forecast/reorder-priority?band=3&limit=5").json()
    assert all(i["priority_band"] == 3 for i in b3["items"])
    assert b3["count"] < NEEDS_REORDER
    assert b3["totals"]["reorder_lines"] == NEEDS_REORDER   # totals stay portfolio-wide


def test_reorder_priority_respects_the_category_filter(client):
    whole = client.get("/forecast/reorder-priority?limit=1").json()
    onco = client.get("/forecast/reorder-priority?Category=Onco Drugs&limit=5").json()
    assert 0 < onco["count"] < whole["count"]
    assert all(i["category"] == "Onco Drugs" for i in onco["items"])
    parts = sum(client.get(f"/forecast/reorder-priority?Category={c}&limit=1").json()["count"]
                for c in da.CATEGORIES)
    assert parts == whole["count"]


def test_existing_fields_keep_their_original_meaning(client):
    body = client.get("/forecast/replenishment-insights").json()
    t = body["totals"]
    # reorder_skus / reorder_now_skus / spectrum are all exactly as before
    assert t["reorder_skus"] == NEEDS_REORDER
    assert t["reorder_now_skus"] == LABELLED_REORDER_NOW
    assert {s["status"]: s["count"] for s in body["spectrum"]} == STATUS_TOTALS
    # and the complete picture arrives as NEW keys alongside them
    assert body["reorder"]["reorder_lines"] == NEEDS_REORDER
    assert sum(b["lines"] for b in body["priority"]) == NEEDS_REORDER
    assert body["priority_queue"][0]["priority_rank"] == 1


def test_reorder_by_status_makes_the_old_gap_visible(client):
    body = client.get("/forecast/replenishment-insights").json()
    assert {s["status"]: s["lines"] for s in body["reorder_by_status"]} == HIDDEN_UNDER


def test_risk_items_priority_mode_is_additive(client):
    old = client.get("/forecast/risk-items?status=Reorder now&limit=5").json()
    assert old["count"] == LABELLED_REORDER_NOW          # unchanged behaviour
    new = client.get("/forecast/risk-items?kind=priority&limit=5").json()
    assert new["count"] == NEEDS_REORDER
    assert new["items"][0]["priority_band"] == 1


# ── rupee-coverage disclosure ────────────────────────────────────────────────
# The reorder queue can only price ~16.5% of its lines, and the reason is specific:
# unit_cost is closing_stock_value / closing_stock, which is undefined once stock hits
# zero — so the UNPRICED lines are very nearly the OUT-OF-STOCK lines (the most urgent
# band). The disclosure used to assert a flat "the rest carry no unit cost in the source
# data", which read as a missing feed and pointed anyone investigating at the wrong fix.
# It now derives the zero-stock / no-cost split from real counts, because that split is
# overwhelming but never exactly total (99.37%–100% depending on the filter).
def test_value_disclosure_reports_the_real_priced_share(client):
    t = client.get("/forecast/reorder-priority").json()["totals"]
    d = t["value_disclosure"]
    assert f"{t['priced_lines']:,}" in d and f"{t['reorder_lines']:,}" in d
    assert f"{t['priced_share_pct']}%" in d
    # counts stay complete even though rupees don't — that promise must survive rewording
    assert "Line and quantity counts are complete." in d


def test_value_disclosure_explains_zero_stock_rather_than_a_missing_feed(client):
    d = client.get("/forecast/reorder-priority").json()["totals"]["value_disclosure"]
    assert "zero stock" in d
    assert "derived from stock on hand" in d


def test_value_disclosure_split_adds_up_and_covers_its_edge_cases():
    from app.api.legacy_kpi import _value_disclosure
    # normal: 15,885 unpriced = 15,878 zero-stock + 7 with no recorded cost
    d = _value_disclosure(3129, 19014, 15878)
    assert "15,885" in d and "15,878" in d and "the other 7" in d
    # every unpriced line is a stock-out -> no dangling ", and 0 with no cost recorded"
    only_zero = _value_disclosure(100, 200, 100)
    assert "All 100" in only_zero and "the other" not in only_zero
    # none are stock-outs -> must NOT claim zero stock is the reason
    none_zero = _value_disclosure(100, 200, 0)
    assert "no recorded cost" in none_zero and "already at zero stock" not in none_zero
    # fully priced -> no caveat about missing rupees at all
    assert _value_disclosure(50, 50, 0).endswith("Every line is priced.")
    # empty selection must not ZeroDivisionError
    assert _value_disclosure(0, 0, 0)
