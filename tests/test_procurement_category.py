# tests/test_procurement_category.py — the material-category cut on PROCUREMENT.
#
# Six of the eight procurement aggregates were groupby'd past material grain, so
# `?Category=` on them used to be an invisible no-op: the response came back identical
# under every bucket while the card above it said "Onco Drugs". app/core/drill_sources.py
# now rebuilds those six from fact_po / fact_grn. Three properties have to hold, and each
# of them is easy to break silently:
#
#   1. RECONCILIATION. Run with no category, every rebuild reproduces its committed
#      parquet row for row and column for column. That is what makes the substitution
#      safe — Rs 649.91 Cr of PO spend is Rs 649.91 Cr whichever path produced it.
#   2. ADDITIVITY. The six buckets partition the portfolio: nothing double-counted,
#      nothing dropped, on every money measure and on the line/quantity counts too.
#   3. VOCABULARY SEPARATION. fact_po and kpi_purchase_value ship their own `category`
#      column which is the PO SPEND taxonomy (~1,360 values like ANTINEOPLASTIC,
#      CAPITALS), NOT our six material buckets. A material filter must never be applied
#      to that column, a spend-taxonomy value must never be accepted as a material
#      bucket, and /kpi/purchase-value under a category must return neither Rs 0 (the
#      old bug) nor the whole portfolio (the bug that would replace it).
#
# Plus the refusal: vendor lead time deliberately does NOT take the cut, and says so
# with a 400 rather than answering unfiltered under a filtered label.
from __future__ import annotations

import pandas as pd
import pytest

from app.core import data_access as da
from app.core import drill_sources as ds

# Verified directly off fact_po.total_value_wo_tax, and equal to kpi_purchase_value's
# own total to the rupee — which is what makes the regrain a substitution rather than a
# re-estimate.
TOTAL_PO_SPEND = 6499128424.0            # Rs 649.9128 Cr
ONCO_PO_SPEND = 2367280144.63            # Rs 236.7280 Cr

# The six buckets' share of PO spend. Unclassified is the LARGEST bucket here (capex,
# services and non-catalogue spend) — the exact opposite of stock, where it is Rs 0 —
# so it must be offered on procurement, never hidden as an empty bucket.
EXPECTED_SPEND_SHARES = {
    "Unclassified": 41.91,
    "Onco Drugs": 36.42,
    "Consumables": 9.48,
    "Other Drugs": 9.21,
    "Non-Medical": 2.67,
    "Lab": 0.30,
}

PROC_TABLES = sorted(ds.PROC_REGRAIN)


# ---------- 1. the rebuild reconciles to the committed parquet ----------

@pytest.mark.parametrize("table", PROC_TABLES)
def test_rebuild_with_no_category_reproduces_the_committed_parquet_exactly(table):
    """The whole safety argument in one assertion.

    If the unfiltered rebuild is not identical to the parquet, then a filtered rebuild
    is measuring something the dashboard never showed, and 'the buckets add up' would be
    adding up to the wrong total.
    """
    built = ds.PROC_REGRAIN[table](None).reset_index(drop=True)
    parquet = da.load(table).reset_index(drop=True)

    assert len(built) == len(parquet), f"{table}: row count moved"
    assert set(parquet.columns) <= set(built.columns), f"{table}: missing columns"

    keys = [c for c in ("plant", "vendor_name", "category", "year", "month")
            if c in parquet.columns]
    a = built[list(parquet.columns)].sort_values(keys).reset_index(drop=True)
    b = parquet.sort_values(keys).reset_index(drop=True)
    for c in parquet.columns:
        if b[c].dtype.kind in "fiu":
            delta = (pd.to_numeric(a[c], errors="coerce").fillna(-9e18)
                     - pd.to_numeric(b[c], errors="coerce").fillna(-9e18)).abs().max()
            assert delta < 1e-6, f"{table}.{c}: max diff {delta}"
        else:
            assert (a[c].astype(str) == b[c].astype(str)).all(), f"{table}.{c}: values differ"


@pytest.mark.parametrize("table", PROC_TABLES)
def test_no_category_never_enters_the_regrain_at_all(table):
    """The unfiltered response must not merely EQUAL today's — it must be the same code.

    regrain() returning None is what sends the request down the original da.load path,
    so an unfiltered dashboard number is produced by identical code reading the identical
    file. Every falsy / "All" spelling has to take that branch.
    """
    for nothing in (None, "", "All", "ALL", "All Categories", "all items"):
        assert ds.regrain(table, da.resolve_category(nothing)) is None


# ---------- 2. the buckets partition the portfolio ----------

@pytest.mark.parametrize("table,measures", [
    ("kpi_purchase_value", ["purchase_value", "purchase_qty", "po_lines"]),
    ("kpi_procurement_variance", ["purchase_value"]),
    ("kpi_vendor_volume", ["vendor_value", "vendor_qty", "po_lines"]),
    ("kpi_purchase_by_location", ["purchase_value", "purchase_qty", "po_lines"]),
    ("kpi_fill_rate", ["ordered_qty", "open_qty"]),
    ("kpi_cycle_time", ["gr_lines"]),
])
def test_the_six_buckets_sum_back_to_the_unfiltered_total(table, measures):
    full = da.load(table)
    parts = [ds.regrain(table, c) for c in da.CATEGORIES]
    for m in measures:
        whole = float(pd.to_numeric(full[m], errors="coerce").sum())
        summed = sum(float(pd.to_numeric(p[m], errors="coerce").sum()) for p in parts)
        assert summed == pytest.approx(whole, rel=1e-9, abs=1e-4), f"{table}.{m}"


def test_po_spend_partitions_across_the_buckets_at_the_verified_shares():
    total = 0.0
    for name, share in EXPECTED_SPEND_SHARES.items():
        v = float(ds.regrain("kpi_purchase_value", name)["purchase_value"].sum())
        assert v / TOTAL_PO_SPEND * 100 == pytest.approx(share, abs=0.01), name
        total += v
    assert total == pytest.approx(TOTAL_PO_SPEND, rel=1e-9)


def test_unclassified_is_the_largest_procurement_bucket_and_must_be_offered():
    """On stock, Unclassified is Rs 0 and the selector rightly hides it. On procurement
    it is Rs 272 Cr of capex/services/non-catalogue spend — the single largest bucket.
    Hiding it here would drop 42% of the money out of a filter built to stop money going
    missing."""
    by = {c: float(ds.regrain("kpi_purchase_value", c)["purchase_value"].sum())
          for c in da.CATEGORIES}
    assert by[da.CATEGORY_UNCLASSIFIED] > 0
    assert max(by, key=by.get) == da.CATEGORY_UNCLASSIFIED


def test_a_bucket_is_a_strict_subset_never_the_whole_portfolio():
    """The failure mode this whole change set exists to remove: a filter that quietly
    hands back everything is indistinguishable from a filter that worked."""
    for table in PROC_TABLES:
        full = len(da.load(table))
        for c in da.CATEGORIES:
            assert 0 < len(ds.regrain(table, c)) <= full, f"{table} / {c}"
        assert any(len(ds.regrain(table, c)) < full for c in da.CATEGORIES), table


def test_an_unrecognised_bucket_yields_nothing_rather_than_everything():
    empty = ds.regrain("kpi_purchase_value", "Not A Real Bucket")
    assert len(empty) == 0


# ---------- 3. the two `category` vocabularies never cross ----------

def test_kpi_purchase_value_category_is_the_po_spend_taxonomy_not_ours():
    pv = da.load("kpi_purchase_value")
    vals = set(pv["category"].dropna().astype(str))
    assert len(vals) > 1000, "expected the ~1,360-value PO spend taxonomy"
    assert "ANTINEOPLASTIC" in vals
    assert not (vals & set(da.CATEGORIES)), "spend taxonomy leaked our material buckets"
    assert da._is_derived_category_col(pv) is False


def test_filter_category_still_refuses_to_touch_the_spend_taxonomy():
    """The guard that stops a material bucket being matched against spend-taxonomy
    values (which returned Rs 0 under every category before it existed)."""
    pv = da.load("kpi_purchase_value")
    assert len(da.filter_category(pv, "Onco Drugs")) == len(pv)


def test_the_regrain_keeps_the_spend_taxonomy_as_the_category_column():
    """A material cut narrows the ROWS; it must never relabel the bars. 'Where spend
    goes' keeps speaking the PO vocabulary it always has."""
    onco = ds.regrain("kpi_purchase_value", "Onco Drugs")
    vals = set(onco["category"].astype(str))
    assert "ANTINEOPLASTIC" in vals
    assert not (vals & set(da.CATEGORIES))
    # ...and the capex/services bucket of the spend taxonomy drops out of an onco cut,
    # which is the proof the filter really is selecting on material.
    assert "CAPITALS" not in vals
    assert "CAPITALS" in set(da.load("kpi_purchase_value")["category"].astype(str))


def test_a_spend_taxonomy_value_is_not_accepted_as_a_material_bucket():
    """The mirror image of the trap: filtering the MATERIAL dimension by a PO-taxonomy
    string must find nothing, not silently pass through as 'no filter'."""
    assert len(ds.regrain("kpi_purchase_value", "ANTINEOPLASTIC")) == 0


def test_regrained_frames_carry_no_derived_category_for_filter_category_to_double_cut():
    """The builders cut on `material_category` themselves and hand back a frame already
    collapsed past it. If a derived category column survived, da.filter_category would
    fire a SECOND time downstream and the two filters could disagree."""
    for table in PROC_TABLES:
        df = ds.regrain(table, "Onco Drugs")
        assert ds.MATERIAL_CATEGORY_COL not in df.columns, table
        if "category" in df.columns:
            assert da._is_derived_category_col(df) is False, table
        # ...so the downstream filter really is a no-op on it
        assert len(da.filter_category(df, "Consumables")) == len(df), table


# ---------- endpoint level ----------

PROC_INSIGHTS = [
    ("/kpi/purchase-value/insights", ("totals", "spend"), TOTAL_PO_SPEND),
    ("/kpi/purchase-by-location/insights", ("totals", "total"), TOTAL_PO_SPEND),
    ("/kpi/vendor-volume-contribution/insights", ("totals", "total"), TOTAL_PO_SPEND),
    ("/kpi/monthly-purchase-value/insights", ("totals", "total"), TOTAL_PO_SPEND),
    ("/portfolio/procurement/overview", ("totals", "spend"), TOTAL_PO_SPEND),
]


def _dig(body, path):
    for k in path:
        body = body[k]
    return body


@pytest.mark.parametrize("url,path,expected", PROC_INSIGHTS)
def test_endpoint_unfiltered_total_is_unchanged(client, url, path, expected):
    assert _dig(client.get(url).json(), path) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("url,path,expected", PROC_INSIGHTS)
def test_endpoint_buckets_sum_back_to_the_unfiltered_total(client, url, path, expected):
    total = 0.0
    for c in da.CATEGORIES:
        r = client.get(url, params={"Category": c})
        assert r.status_code == 200, f"{url} / {c}: {r.text[:200]}"
        total += _dig(r.json(), path)
    assert total == pytest.approx(expected, rel=1e-9)


def test_purchase_value_under_a_category_is_neither_zero_nor_the_whole_portfolio(client):
    """Both historical failure modes at once. Before the spend-taxonomy guard,
    /kpi/purchase-value?Category=... returned Rs 0; after it, the request was refused and
    the FULL Rs 649.91 Cr came back under an 'Onco Drugs' label, contradicting
    monthly-purchase-value which reported Rs 236.73 Cr for the same filter."""
    body = client.get("/kpi/purchase-value/insights",
                      params={"Category": "Onco Drugs"}).json()
    spend = body["totals"]["spend"]
    assert spend == pytest.approx(ONCO_PO_SPEND, rel=1e-9)
    assert spend != 0.0
    assert spend < TOTAL_PO_SPEND

    # ...and the two procurement money metrics now agree instead of contradicting.
    monthly = client.get("/kpi/monthly-purchase-value/insights",
                         params={"Category": "Onco Drugs"}).json()["totals"]["total"]
    assert monthly == pytest.approx(spend, rel=1e-9)


def test_summary_endpoint_sums_back_across_buckets(client):
    whole = client.get("/kpi/purchase-value/summary").json()["purchase_value"]["sum"]
    parts = sum(client.get("/kpi/purchase-value/summary", params={"Category": c})
                .json()["purchase_value"]["sum"] for c in da.CATEGORIES)
    assert whole == pytest.approx(TOTAL_PO_SPEND, rel=1e-9)
    assert parts == pytest.approx(whole, rel=1e-9)


def test_a_fully_null_measure_summarises_to_null_not_zero(client):
    """fact_grn records no PR→GR turnaround at all on onco receipts. `null` says 'no
    data in this bucket'; 0.0 would say 'requisitioned and received the same day'."""
    body = client.get("/kpi/procurement-cycle-time/summary",
                      params={"Category": "Onco Drugs"})
    assert body.status_code == 200, body.text[:300]
    assert body.json()["avg_pr_to_gr_tat"]["mean"] is None
    assert body.json()["avg_po_to_gr_tat"]["mean"] is not None


def test_cycle_time_moves_by_bucket_and_is_the_exact_mean_of_that_bucket(client):
    """Not a fabricated number: the regrain rebuilds the mean from fact_grn's TAT sums
    and counts, so it equals what transforms.py's own mean() would produce on the cut."""
    per_cat = {}
    for c in da.CATEGORIES:
        per_cat[c] = client.get("/kpi/procurement-cycle-time/insights",
                                params={"Category": c}).json()["totals"]["avg_po"]
    assert per_cat["Onco Drugs"] < per_cat["Lab"]
    assert len(set(round(v, 4) for v in per_cat.values())) == len(per_cat)

    grn = da.load("fact_grn")
    tat = pd.to_numeric(grn["po_to_gr_tat"], errors="coerce")
    grn = grn.assign(_t=tat.mask((tat < 0) | (tat > 365)))
    # The endpoint reports over its own six-month reporting window (legacy_kpi._PROC_WINDOW),
    # so the ground truth has to be cut to the same window — otherwise this would be
    # comparing two different questions and calling the difference a bug.
    from app.api.legacy_kpi import _PROC_WINDOW
    win = grn[[(int(y), str(m)) in _PROC_WINDOW for y, m in zip(grn["year"], grn["month"])]]
    for c in ("Onco Drugs", "Lab", "Consumables"):
        truth = float(win[win["category"] == c]["_t"].mean())
        assert per_cat[c] == pytest.approx(truth, abs=0.02), c


def test_vendor_concentration_is_recomputed_within_the_bucket(client):
    """Shares must be shares OF THE CUT. Carrying the portfolio-wide share into a
    filtered view would leave a column that no longer sums to 100."""
    onco = client.get("/kpi/vendor-volume-contribution/insights",
                      params={"Category": "Onco Drugs"}).json()
    allc = client.get("/kpi/vendor-volume-contribution/insights").json()
    assert onco["totals"]["top1"] > allc["totals"]["top1"]      # 94.9% vs 45.8%
    assert onco["totals"]["hhi"] > allc["totals"]["hhi"]        # 9,004 vs 2,140
    assert onco["totals"]["vendors"] < allc["totals"]["vendors"]
    assert onco["vendors"][0]["share"] == pytest.approx(onco["totals"]["top1"], abs=0.01)
    assert sum(v["share"] for v in onco["vendors"]) <= 100.0 + 1e-6


def test_fill_rate_discloses_the_service_po_cap_instead_of_printing_a_confident_100(client):
    """2,557 fact_po lines carry a NEGATIVE open_qty; all of them are Unclassified and
    2,555 are service/blanket POs. The raw ratio for that bucket is 168.6%. The clamp
    already existed — printing 100% and saying nothing is what would be misleading."""
    unc = client.get("/kpi/fill-rate/insights",
                     params={"Category": da.CATEGORY_UNCLASSIFIED}).json()
    assert unc["totals"]["overall"] == pytest.approx(100.0)
    assert unc["totals"]["overall_raw"] > 100.0
    assert unc["totals"]["capped"] is True
    assert unc["note"] and "service" in unc["note"].lower()

    # The five real material buckets are CLEANER than the unfiltered figure, not dirtier.
    for c in ("Onco Drugs", "Other Drugs", "Consumables", "Lab", "Non-Medical"):
        body = client.get("/kpi/fill-rate/insights", params={"Category": c}).json()
        assert body["totals"]["capped"] is False, c
        assert 0.0 <= body["totals"]["overall_raw"] <= 100.0, c
        assert body["note"] is None, c


def test_unfiltered_responses_gain_no_new_keys(client):
    """Advisory fields are attached ONLY when a category is applied. The regression gate
    compares whole JSON documents, so a new key on the default path is a breakage."""
    for url, extra in (("/kpi/fill-rate/insights", ("category", "note")),
                       ("/portfolio/procurement/overview", ("category", "category_notes"))):
        body = client.get(url).json()
        for k in extra:
            assert k not in body, f"{url} gained '{k}' with no Category applied"
        assert "capped" not in body["totals"]
        assert "overall_raw" not in body["totals"]
        assert "completion_raw" not in body["totals"]


# ---------- the refusal ----------

VLT_ROUTES = ["/kpi/vendor-lead-time", "/kpi/vendor-lead-time/summary",
              "/kpi/vendor-lead-time/table", "/kpi/vendor-lead-time/insights",
              "/drill/top-items?kpi=vendor-lead-time&by=vendor"]


@pytest.mark.parametrize("url", VLT_ROUTES)
def test_vendor_lead_time_refuses_a_category_instead_of_answering_unfiltered(client, url):
    """The one procurement metric that cannot honestly take the cut. Returning the whole
    portfolio under an 'Onco Drugs' label is the single outcome worse than refusing: the
    reader cannot tell a filter that did nothing from a filter that found everything."""
    sep = "&" if "?" in url else "?"
    r = client.get(f"{url}{sep}Category=Onco%20Drugs")
    assert r.status_code == 400, f"{url} answered {r.status_code}, not a refusal"
    assert "procurement-cycle-time" in r.json()["detail"], "must name the alternative"


@pytest.mark.parametrize("url", VLT_ROUTES)
def test_vendor_lead_time_is_untouched_without_a_category(client, url):
    assert client.get(url).status_code == 200


def test_vendor_lead_time_stays_out_of_the_regrain_map():
    assert "kpi_vendor_lead_time" not in ds.PROC_REGRAIN
    assert "kpi_vendor_lead_time" in ds.CATEGORY_UNSUPPORTED


# ---------- the published contract ----------

def test_category_support_contract_matches_what_the_endpoints_actually_do(client):
    """A card wires its filter off this endpoint, so a lie here becomes a dead control."""
    rows = {r["key"]: r for r in client.get("/meta/category-support").json()["kpis"]}
    procurement = ["purchase-value", "monthly-purchase-value", "procurement-variance",
                   "vendor-volume-contribution", "purchase-by-location",
                   "procurement-cycle-time", "vendor-lead-time", "fill-rate"]
    assert set(procurement) <= set(rows)

    assert rows["vendor-lead-time"]["supported"] is False
    assert rows["vendor-lead-time"]["reason"]
    for k in procurement:
        if k == "vendor-lead-time":
            continue
        assert rows[k]["supported"] is True, k
        assert rows[k]["how"] in ("native", "regrain"), k

    # ...and every claim of support really does narrow the number.
    for k in procurement:
        r = client.get(f"/kpi/{k}/summary", params={"Category": "Onco Drugs"})
        if not rows[k]["supported"]:
            assert r.status_code == 400, k
            continue
        assert r.status_code == 200, k
        full = client.get(f"/kpi/{k}/summary").json()
        assert r.json()["row_count"] < full["row_count"], f"{k} did not narrow"


def test_the_drill_narrows_with_the_bar_it_was_launched_from(client):
    """A drill must mirror its chart. If the card says Rs 236.7 Cr of onco spend, the
    panel inside it cannot total Rs 649.9 Cr."""
    for kpi in ("purchase-value", "vendor-volume-contribution", "purchase-by-location",
                "procurement-variance"):
        # One query string, not url + params=: httpx REPLACES an existing query when
        # `params` is given, which would silently drop `kpi` and test a 422 instead.
        body = client.get(f"/drill/top-items?kpi={kpi}&by=material"
                          f"&Category=Onco%20Drugs").json()
        assert body["grand_total"] == pytest.approx(ONCO_PO_SPEND, rel=1e-6), kpi
        assert client.get(f"/drill/matrix?kpi={kpi}").json()["kpis"][0]["respects_category"]
