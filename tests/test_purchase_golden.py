# tests/test_purchase_golden.py — HCG's own out-of-formulary figures, on the committed purchase tables.
#
# HCG's monthly OOF workbooks ('Out of Formulary Spend Analysis June 26 WS.xlsb', 'July 26 WS',
# 'Brand Margin loss Spend anlayiss August 26') end in a Summary sheet: per centre, the margin
# the OOF brands earned, the margin HCG's suggested brands would have earned, and the
# difference ('Margin Value Differnce'). HCG's spend email quotes the KR (HC05) row. This is the
# one place those figures are written down: the build computes everything from the GRN lines,
# the Item Masters and HCG's worksheets; this test sets the committed tables beside HCG.
#
# It also pins the simulation the build records in purchase_validation.json: what a month
# without HCG's worksheet would show (HCG's pairs from earlier worksheets, HCG's master, our
# rule), as found when the rule was chosen (Jun +7.4%, Jul -1.7%, Aug -0.7% vs HCG).
#
# Runs only once `python -m app.etl.purchase` has published into settings.PURCHASE_DATA_DIR.
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import settings
from app.etl.purchase import build as B
from app.etl.purchase import config as C
from app.etl.purchase import oof as O

DATA = Path(settings.PURCHASE_DATA_DIR)
pytestmark = pytest.mark.skipif(not (DATA / f"{C.OOF}.parquet").exists(),
                                reason="purchase tables not built — run python -m app.etl.purchase")

# HCG's Summary: the net loss of all centres, and how many centres it lists.
HCG_TOTALS = {"2026-06": (2_837_256.29, 20), "2026-07": (2_661_388.30, 21), "2026-08": (3_243_199.13, 20)}
# HCG's spend email, KR out of formulary: actual margin, proposed margin, loss (rupees, rounded).
KR_EMAIL = {"2026-06": (173_895, 949_266, 775_371), "2026-07": (113_341, 911_435, 798_094),
            "2026-08": (492_782, 1_816_364, 1_323_583)}
# Our lines and how HCG's sheet leaves them: lines, parked, without a brand.
LINES = {"2026-06": (765, 0, 4), "2026-07": (815, 12, 0), "2026-08": (860, 0, 19)}
# The simulation without HCG's worksheet: ours - HCG, % of HCG's loss (all centres, and KR).
SIMULATION = {"2026-06": (7.44, 10.2), "2026-07": (-1.66, 0.07), "2026-08": (-0.72, -0.19)}
PAISA = 0.005


@pytest.fixture(scope="module")
def oof():
    return pd.read_parquet(DATA / f"{C.OOF}.parquet")


@pytest.fixture(scope="module")
def hcg():
    return pd.read_parquet(DATA / f"{C.HCG_SUMMARY}.parquet")


@pytest.fixture(scope="module")
def report():
    return json.loads((DATA / C.VALIDATION_FILE).read_text())


@pytest.fixture()
def months(oof):
    built = set(oof["period"])
    missing = sorted(set(HCG_TOTALS) - built)
    if missing:
        pytest.skip(f"not built: {', '.join(missing)}")
    return sorted(HCG_TOTALS)


def test_hcgs_summary_is_read_whole(hcg, months):
    for p in months:
        s = hcg[hcg["period"] == p]
        total, centres = HCG_TOTALS[p]
        assert s["loss"].sum() == pytest.approx(total, abs=PAISA), p
        assert len(s) == centres and s["plant"].is_unique, p
        assert (s["loss"] - (s["sug_margin"] - s["used_margin"])).abs().max() < PAISA, p


def test_every_centre_equals_hcgs_summary_to_the_paisa(oof, hcg, months):
    for p in months:
        o = oof[oof["period"] == p]
        counted = o[o["status"] == O.COUNTED]
        ours = pd.DataFrame({
            "used_pp": counted.groupby("plant")["total_pp"].sum(),
            "used_mrp": counted.groupby("plant")["total_mrp"].sum(),
            "used_margin": o.groupby("plant")["actual_margin"].sum(),
            "sug_pp": o.groupby("plant")["sug_total_pp"].sum(),
            "sug_mrp": o.groupby("plant")["sug_total_mrp"].sum(),
            "sug_margin": o.groupby("plant")["suggested_margin"].sum(),
            "loss": o.groupby("plant")["margin_loss"].sum()}).fillna(0.0)
        theirs = hcg[hcg["period"] == p].set_index("plant")[list(ours.columns)]
        ours = ours.reindex(theirs.index.union(ours.index)).fillna(0.0)
        theirs = theirs.reindex(ours.index).fillna(0.0)
        worst = (ours - theirs).abs().max()
        assert (worst < PAISA).all(), f"{p}: {worst[worst >= PAISA].to_dict()}"
        assert o["margin_loss"].sum() == pytest.approx(HCG_TOTALS[p][0], abs=PAISA)


def test_kr_equals_hcgs_email(oof, months):
    for p in months:
        kr = oof[(oof["period"] == p) & (oof["plant"] == C.KR_PLANT)]
        got = (kr["actual_margin"].sum(), kr["suggested_margin"].sum(), kr["margin_loss"].sum())
        for g, want in zip(got, KR_EMAIL[p]):
            assert abs(g - want) <= O.MAX_HCG_DIFF, (p, got, KR_EMAIL[p])
    aug = oof[(oof["period"] == "2026-08") & (oof["plant"] == C.KR_PLANT)]
    assert 100 * aug["margin_loss"].sum() / aug["suggested_margin"].sum() == pytest.approx(72.87, abs=0.01)


def test_hcg_months_follow_hcgs_sheet_alone(oof, months):
    for p in months:
        o = oof[oof["period"] == p]
        assert set(o["suggestion_source"]) <= {O.HCG_SHEET, O.NONE}, p
        lines, parked, none = LINES[p]
        by = o.groupby("status")["lines"].sum()
        assert (o["lines"].sum(), by.get(O.PARKED, 0), by.get(O.NO_ALTERNATIVE, 0)) == (lines, parked, none), p
        assert set(o["status"]) <= set(O.STATUSES) and set(o["segment"]) <= {"onco", "non-onco"}
        off = o[o["status"] != O.COUNTED]
        assert (off[["actual_margin", "suggested_margin", "margin_loss"]] == 0).all().all(), p
        assert (o["gain_loss"] + o["better_loss"] - o["margin_loss"]).abs().max() < 1e-6, p
        assert o["better_loss"].sum() < 0 < o["gain_loss"].sum(), p      # HCG nets both


def test_the_simulation_without_hcgs_sheet_is_recorded(report, months):
    for p in months:
        sim = report["months"][p]["oof"]["simulation"]
        want, want_kr = SIMULATION[p]
        assert sim["diff_pct"] == pytest.approx(want, abs=0.05), p
        assert sim["kr"]["diff_vs_email_pct"] == pytest.approx(want_kr, abs=0.05), p
        assert sim["pair_months"] == [m for m in reversed(months) if m < p], p
        assert sim["brands"]["same_brand_pct"] > 85, p
    assert report["months"]["2026-08"]["oof"]["hcg"]["max_centre_diff"] < O.MAX_HCG_DIFF


def test_the_published_tables_are_aggregates(oof, report):
    for name in (C.OOF, C.HCG_SUMMARY, C.HCG_PAIRS, C.HCG_PARKED):
        df = pd.read_parquet(DATA / f"{name}.parquet")
        B.check_columns(name, df)
        assert not {c.upper() for c in df.columns} & {"GR_QTY", "GR_DATE", "PO_DATE"}, name
    assert list(oof.columns) == O.OOF_COLUMNS
    assert len(oof) < sum(v[0] for v in LINES.values()) and oof["lines"].min() >= 1
    assert report["tables"][C.OOF]["rows"] == len(oof)
